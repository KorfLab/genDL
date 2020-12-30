import torch
from torch import nn
#import torch.nn as nn
import torch.nn.functional as F #contains the convolutional functions such as conv1d
import torch.optim as optim
import numpy as np
import argparse
import sys
import gzip
import random
import pickle
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from sklearn.model_selection import train_test_split

from dynamic_model import DynamicNet

from torch.utils.tensorboard import SummaryWriter
import torchvision

from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import random_split


#python3 -m tensorboard.main --logdir=runs --bind_all

parser = argparse.ArgumentParser(description='Pytorch')
parser.add_argument('--path', required=True, type=str,
	metavar='<path>', help='path to csv file')
parser.add_argument('--split', required=False, type=float, default = 0.2,
	metavar = '<float>', help = 'split size')
parser.add_argument('--epoch', required=False, type=int, default = 10,
	metavar='<int>', help='number of epochs')
parser.add_argument('--batch', required=False, type=int, default = 1,
	metavar='<int>', help='batch size')

arg = parser.parse_args()


class CSVDataset(Dataset):
	# load the dataset
	def __init__(self, path):
		# load the csv file as a dataframe
		df = read_csv(path, header=None)
		# store the inputs and outputs
		self.X = df.values[:, :-1]
		self.y = df.values[:, -1]
		# ensure input data is floats
		self.X = self.X.astype('float32')
		# label encode target and ensure the values are floats
		self.y = LabelEncoder().fit_transform(self.y)
		self.y = self.y.astype('float32')
		self.y = self.y.reshape((len(self.y), 1))
	# number of rows in the dataset
	def __len__(self):
		return len(self.X)

	# get a row at an index
	def __getitem__(self, idx):
		return [self.X[idx], self.y[idx]]

	# get indexes for train and test rows
	def get_splits(self, n_test=0.33):
		# determine sizes
		test_size = round(n_test * len(self.X))
		train_size = len(self.X) - test_size
		# calculate the split
		return random_split(self, [train_size, test_size])

dataset = CSVDataset(arg.path)
train_s, test_s = dataset.get_splits()
train_seq = torch.utils.data.DataLoader(train_s, batch_size = arg.batch, shuffle = True)
test_seq = torch.utils.data.DataLoader(test_s, batch_size = 1, shuffle = False)


class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		#self.conv1 = nn.Conv1d(in_channels = 42, out_channels= 84 , kernel_size = 2)
		#self.maxpool = nn.MaxPool2d(42, stride=2)
		self.fc1 = nn.Linear(in_features=168, out_features=84)
		self.act1 = nn.ReLU()
		self.fc2 = nn.Linear(in_features=84, out_features=42)
		self.act2 = nn.ReLU()
		#self.conv2 = nn.Conv2d(in_channels = 42, out_channels = 42, kernel_size = 3)
		self.fc3 = nn.Linear(in_features=42, out_features=21)
		self.act3 = nn.ReLU()
		self.fc4 = nn.Linear(in_features=21, out_features=12)
		self.act4 = nn.ReLU()
		self.fc5 = nn.Linear(in_features=12, out_features=10)
		self.act5 = nn.ReLU()
		self.fc6 = nn.Linear(in_features=10, out_features=5)
		self.act6 = nn.ReLU()
		self.fc7 = nn.Linear(in_features=5, out_features=2)
		self.act7 = nn.ReLU()
		self.out = nn.Linear(in_features=2, out_features=1) ###
		self.actout = nn.Sigmoid()


	def forward(self, t):
		#t = t.flatten(start_dim=1)
		t = self.fc1(t)
		t = self.act1(t)
		t = self.fc2(t)
		t = self.act2(t)
		t = self.fc3(t)
		t = self.act3(t)
		t = self.fc4(t)
		t = self.act4(t)
		t = self.fc5(t)
		t = self.act5(t)
		t = self.fc6(t)
		t = self.act6(t)
		t = self.fc7(t)
		t = self.act7(t)
		t = self.out(t)
		t = self.actout(t)
		return(t)


#model9 = DynamicNet(len(seqs[0])*4, [21, 10], [nn.ReLU, nn.ReLU])
model8 = DynamicNet(len(dataset[0][0]), [84, 42, 21, 10, 5, 2], [nn.ReLU, nn.ReLU, nn.ReLU, nn.ReLU, nn.ReLU, nn.ReLU])

#model8 = Net()
nets = []
nets.append(model8)

###terminal graph


#tensorboard to draw the model
#tb = SummaryWriter()


def get_num_correct(preds, labels):
	return preds.argmax(dim=1).eq(labels).sum().item()

accuracy = []
for net in nets:
	#defining a loss function and optimizer
	criterion = nn.BCELoss()
	#optimizer = optim.SGD(net.parameters(), lr=1e-4, momentum=0.9, weight_decay = 1e-4)
	optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay = 1e-4)
	##try a different optimizer
	#'''
	for epoch in range(arg.epoch):  # loop over the dataset multiple times
		total_correct = 0
		total_loss = 0
		running_loss = 0.0
		for i, (inputs, targets) in enumerate(train_seq):
			#print(seq)
			#print(label)
			#sys.exit()
			# zero the parameter gradients
			optimizer.zero_grad()
			outputs = net(inputs)
			#print(outputs, label)
			#print(outputs, len(outputs))
			#print(label)
			#sys.exit()
			loss = criterion(outputs, targets)

			#writing to scalar
			#tb.add_scalar("Loss/train", loss, epoch)
			###
			loss.backward()
			optimizer.step()

			##accuracy and loss in training set
			total_loss += loss.item()
			#if torch.round(outputs) == label:
				#total_correct += 1
				#total_correct += get_num_correct(outputs, label)
			# print statistics
			running_loss += loss.item()
			if i % 100 == 0:    # print every 2000 mini-batches
				#print(epoch, loss.data[0])
				print('[%d, %5d] loss: %.3f' % (epoch + 1, i, running_loss / 100))
				###recording training loss form each epoch into the writer
				running_loss = 0.0
	#'''
		#print(total_correct/len(seq_train))
		#sys.exit()
		#print(total_correct)
		#print(total_correct/len(seq_train))
		#tb.add_scalar("Loss", total_loss, epoch)
		#tb.add_scalar("Correct", total_correct, epoch)
		#tb.add_scalar("Accuracy", total_correct/len(seq_train), epoch)
			#changing the batch size

		#for name, weight in model8.named_parameters():
			#tb.add_histogram(name, weight, epoch)
			#tb.add_histogram(f'{name}.grad', weight.grad, epoch)
	#tb.flush()
	print('Finished Training')
	correct = 0
	total = 0
	with torch.no_grad():
		for i, (inputs, targets) in enumerate(test_seq):
			outputs = net(inputs)
			predicted = torch.round(outputs)
			total += targets.size(0)
			if predicted == targets:
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
