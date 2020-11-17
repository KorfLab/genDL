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
from torchvision import transforms, utils
from sklearn.model_selection import train_test_split

from dynamic_model import DynamicNet

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


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        #self.conv1 = nn.Conv1d(in_channels = 42, out_channels= 84 , kernel_size = 2)
        #self.maxpool = nn.MaxPool2d(42, stride=2)
        self.fc1 = nn.Linear(in_features=168, out_features=84)
        self.fc2 = nn.Linear(in_features=84, out_features=42)
        #self.conv2 = nn.Conv2d(in_channels = 42, out_channels = 42, kernel_size = 3)
        self.fc3 = nn.Linear(in_features=42, out_features=24)
        self.fc4 = nn.Linear(in_features=24, out_features=12)
        self.out = nn.Linear(in_features=12, out_features=1) ###


    def forward(self, t):
        #t = F.relu(self.conv1(t))

        #t = t.flatten(start_dim=1)
        #t = F.max_pool2d(t, kernel_size=2, stride=2)

        #t = F.relu(self.conv2(t))
        #t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = t.flatten(start_dim=1)
        t = F.elu(self.fc1(t))
        t = F.dropout(t, training=self.training)
        t = F.elu(self.fc2(t))
        t = F.dropout(t, training=self.training)
        #t = F.elu(self.conv2())
        t = torch.tanh(self.fc3(t))
        t = F.dropout(t, training=self.training)
        t = torch.tanh(self.fc4(t))
        t = F.dropout(t, training=self.training)
        t = torch.sigmoid(self.out(t))
        #t = self.out(t)

        return (t)


model1 = DynamicNet(((len(seqs[0]))*4), [42, 21, 12, 5], [nn.ReLU, nn.Tanh, nn.Tanh, nn.Tanh])
model2 = DynamicNet(((len(seqs[0]))*4), [336, 168, 42, 21], [nn.Tanh, nn.Tanh, nn.Tanh, nn.Tanh])

nets = []
nets.append(model1)
nets.append(model2)

#training the network
#print(len(train_seqs))
#print(len(train_labels))
accuracy = []
for net in nets:

    #defining a loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay = 1e-4)
    for epoch in range(arg.epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(zip(train_seqs, train_labels), 0):
            seq, label = data
            # zero the parameter gradients
            optimizer.zero_grad()
            seq = seq.float()
            label = label.float()
            outputs = net(seq)
            #print(outputs, label)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 0:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i, running_loss / 100))
                running_loss = 0.0

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
            #torch.mac() = 0 always
            total += test_label.size(0)
            if predicted == test_label:
                correct += 1
    accuracy.append((correct/total))

for i in range(len(accuracy)):
    print(f'Accuracy of model {i}: {accuracy[i]:.4f}')
#print(f'Accuracy of the network on the test sequences: {(correct/total):.4f}')

###checking on the train - round works, while max does not
###how the network is being initialized
###drop, regular, learning rate
