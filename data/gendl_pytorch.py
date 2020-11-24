import torch
import torch.nn as nn
import torch.nn.functional as F #contains the convolutional functions such as conv1d
import torch.optim as optim
import numpy as np
import argparse
import sys
import gzip
import random
import pickle
from torch.utils.data import Dataset, DataLoader
#from torchvision import transforms, utils
from sklearn.model_selection import train_test_split

from dynamic_model import DynamicNet

#from torch.utils.tensorboard import SummaryWriter
#import torchvision


parser = argparse.ArgumentParser(description='Pytorch')
parser.add_argument('--true', required=True, type=str,
	metavar='<path>', help='path to true pickled file')
parser.add_argument('--fake', required=True, type=str,
	metavar='<path>', help='path to true pickled file')
parser.add_argument('--epoch', required=False, type=int, default = 10,
	metavar='<int>', help='number of epochs')
parser.add_argument('--batch', required=False, type=int, default = 1,
	metavar='<int>', help='batch size')

arg = parser.parse_args()

#unpacking one hot encoding
true = pickle.load(open(arg.true, "rb"))
fake = pickle.load(open(arg.fake, "rb"))
seqs = np.concatenate((true, fake), axis = 0)

#creating target labels
truelabel = ([0]*len(true))
fakelabel = ([1]*len(fake))
labels = np.concatenate((truelabel, fakelabel), axis=0)

#creating validation set
seq_train, seq_val, label_train, label_val = train_test_split(seqs, labels, test_size = 0.2)
#converting seqs into torch format
print(seq_train.shape, seq_val.shape, label_train.shape, label_val.shape)

#loading data
train_seqs = torch.utils.data.DataLoader(seq_train, batch_size = arg.batch, shuffle = False)
train_labels = torch.utils.data.DataLoader(label_train, batch_size = arg.batch, shuffle = False)

val_seqs = torch.utils.data.DataLoader(seq_val, batch_size = 1, shuffle = False)
val_labels = torch.utils.data.DataLoader(label_val, batch_size = 1, shuffle = False)


model9 = DynamicNet(len(seqs[0])*4, [21, 10], [nn.ReLU, nn.ReLU])
nets = []
nets.append(model9)

#tensorboard to draw the model
#tb = SummaryWriter()

#training the network
#print(len(train_seqs))
#print(len(train_labels))

def get_num_correct(preds, labels):
	return preds.argmax(dim=1).eq(labels).sum().item()

accuracy = []
for net in nets:
	#defining a loss function and optimizer
	criterion = nn.BCELoss()
	#optimizer = optim.SGD(net.parameters(), lr=1e-4, momentum=0.9, weight_decay = 1e-4)
	optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay = 1e-4)
	##try a different optimizer
	for epoch in range(arg.epoch):  # loop over the dataset multiple times
		total_correct = 0
		total_loss = 0
		running_loss = 0.0
		for i, data in enumerate(zip(train_seqs, train_labels), 0):
			seq, label = data
			# zero the parameter gradients
			optimizer.zero_grad()
			seq = seq.float()
			label = label.float()
			label = label.unsqueeze(1)
			outputs = net(seq)
			#print(outputs, label)
			#print(outputs, len(outputs))
			#print(label)
			#sys.exit()
			loss = criterion(outputs, label)

			#writing to scalar
			#tb.add_scalar("Loss/train", loss, epoch)
			###
			loss.backward()
			optimizer.step()

			##accuracy and loss in training set
			total_correct += get_num_correct(outputs, label)
			total_loss +=loss.item()

			# print statistics
			running_loss += loss.item()
			if i % 100 == 0:    # print every 2000 mini-batches
				print('[%d, %5d] loss: %.3f' % (epoch + 1, i, running_loss / 100))
				###recording training loss form each epoch into the writer
				running_loss = 0.0
		#print(total_correct/len(seq_train))
		#sys.exit()
		#tb.add_scalar("Loss", total_loss, epoch)
		#tb.add_scalar("Correct", total_correct, epoch)
		#tb.add_scalar("Accuracy", total_correct/len(seq_train), epoch)
			#changing the batch size

		for name, weight in model9.named_parameters():
			#tb.add_histogram(name, weight, epoch)
			#tb.add_histogram(f'{name}.grad', weight.grad, epoch)
	#tb.flush()
	print('Finished Training')

	correct = 0
	total = 0
	with torch.no_grad():
		for test_seq, test_label in zip(val_seqs, val_labels):
			#test_seq = test_seq.clone().detach()
			#test_seq = torch.tensor(test_seq, dtype=torch.float32)
			test_seq = test_seq.float()
			outputs = net(test_seq)
			predicted = torch.round(outputs)
			#torch.max() = 0 always
			total += test_label.size(0)
			if predicted == test_label:
				correct += 1

			#recording loss and accuracy from the test run into the writer
	accuracy.append((correct/total))

#tb.close()


for i in range(len(accuracy)):
	print(f'Accuracy of model {i+1}: {accuracy[i]:.4f}')

###graphic of the network (weights and bias for each node)

#print(f'Accuracy of the network on the test sequences: {(correct/total):.4f}')

###checking on the train - round works, while max does not
###how the network is being initialized
###drop, regular, learning rate
