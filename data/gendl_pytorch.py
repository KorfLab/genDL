import torch
import torch.nn as nn
import torch.nn.functional as F #contains the convolutional functions such as conv1d
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys
import gzip
import random
import pickle
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from sklearn.model_selection import train_test_split

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

'''
#in case if I were to switch to pandas df
def get_seqs(file, limit, start, end):
    seqs = []
    with gzip.open(file, mode='rt') as fp:
        lines = fp.read().splitlines()
        random.shuffle(lines)
        for i in range(limit):
            seqs.append(lines[i][start:end])

    dup = set(seqs)
    #print('no duplicates:', len(dup), 'with duplicates:', len(seqs))

    return seqs
'''
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
train_seqs = torch.utils.data.DataLoader(seq_train, batch_size = 5, shuffle = False)
train_labels = torch.utils.data.DataLoader(label_train, batch_size = 5, shuffle = False)

val_seqs = torch.utils.data.DataLoader(seq_val, batch_size = 1, shuffle = False)
val_labels = torch.utils.data.DataLoader(label_val, batch_size = 1, shuffle = False)

#classes = ('true', 'fake')

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        #self.conv1 = nn.Conv2d(in_channels = 1, out_channels= 6, kernel_size = 5)
        #self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5)
        self.fc1 = nn.Linear(in_features=168, out_features=84)
        self.fc2 = nn.Linear(in_features=84, out_features=42)
        self.fc3 = nn.Linear(in_features=42, out_features=24)
        self.fc4 = nn.Linear(in_features=24, out_features=12)
        self.out = nn.Linear(in_features=12, out_features=1) ###
        ###dropout
        ###implement number of layers and types of layers
        #optimize the way to search the models with the best performance


    def forward(self, t):
        #t = F.relu(self.conv1(t))
        #t = F.max_pool2d(t, kernel_size=2, stride=2)

        #t = F.relu(self.conv2(t))
        #t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = t.flatten(start_dim=1)
        t = F.elu(self.fc1(t))
        t = F.dropout(t, training=self.training)
        t = F.elu(self.fc2(t))
        t = F.dropout(t, training=self.training)
        t = torch.tanh(self.fc3(t))
        t = F.dropout(t, training=self.training)
        t = torch.tanh(self.fc4(t))
        t = F.dropout(t, training=self.training)
        t = torch.sigmoid(self.out(t))
        #t = self.out(t)

        return (t)


net = Net()
#defining a loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay = 1e-4)
#optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay = 0.5)

#training the network
#print(len(train_seqs))
#print(len(train_labels))
for epoch in range(10):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(zip(train_seqs, train_labels), 0):
        seq, label = data
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        #seq = torch.tensor(seq, dtype=torch.float32)
        seq = seq.float()
        #seq = seq.clone().detach()
        label = label.float()
        outputs = net(seq)
        #print(outputs, label)
        #sys.exit()
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 0:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')

#testing on one seq
'''
PATH = './practice.pth'
torch.save(net.state_dict(), PATH)

seq_iter = iter(val_seqs)
iter_seq = (seq_iter).next()
iter_seq = torch.tensor(iter_seq, dtype=torch.float32)

label_iter = iter(val_labels)
iter_label = (label_iter).next()
iter_label = torch.tensor(iter_label, dtype=torch.float32)
print(iter_label)

net = Net()
net.load_state_dict(torch.load(PATH))

output = net(iter_seq)
_, predicted = torch.max(output, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(1)))
'''
correct = 0
total = 0
with torch.no_grad():
    for test_seq, test_label in zip(val_seqs, val_labels):
        #test_seq = test_seq.clone().detach()
        #test_seq = torch.tensor(test_seq, dtype=torch.float32)
        test_seq = test_seq.float()
        #print(test_seq.shape)
        #sys.exit()
        outputs = net(test_seq)
        #print(outputs)
        #sys.exit()
        #_, predicted = torch.max(outputs, 1)
        #print(predicted)

        predicted = torch.round(outputs) ### max always gives me 0 no matter what
        total += test_label.size(0)
        #print(predicted, test_label)
        #sys.exit()
        if predicted == test_label:
            correct += 1
print(f'Accuracy of the network on the test sequences: {(correct/total):.4f}')

###checking on the train - round works, while max does not
###how the network is being initialized
###drop, regular, learning rate
'''
def pytorch_classification(trues, fakes, epochs, batch):

    true_train, true_test, fake_train, fake_test = train_test_split(trues, fakes, test_size = 0.2)

    true_seqs = torch_flatten(true_train)
    fake_seqs = torch_flatten(fake_train)
    seqs = np.concatenate((true_seqs, fake_seqs), axis=0)
    print(seqs.shape)
    sys.exit()

    true_size = len(true_train)
    fake_size = len(fake_train)

    #unpacking one hot encoding
    true = pickle.load(open(trues, "rb"))
    fake = pickle.load(open(fakes, "rb"))
    seqs = np.concatenate((true, fake), axis=0)
    #print(true.shape)
    print(seqs.shape)

    #creating target labels
    truelabel = np.array([[0]]*len(true), dtype=np.uint8)
    fakelabel = np.array([[1]]*len(fake), dtype=np.uint8)
    labels = np.concatenate((truelabel, fakelabel), axis=0)

    #creating validation set
    train_seqs, val_seqs, train_labels, val_labels = train_test_split(seqs, labels, test_size = 0.2)
    print(train_seqs.shape, val_seqs.shape, train_labels.shape, val_labels.shape)
    print(train_seqs.shape[0])


    #converting seqs into torch format
    train_seqs = train_seqs.reshape(train_seqs.shape[0], 1, train_seqs.shape[1], train_seqs.shape[2])
    train_seqs = torch.from_numpy(train_seqs)

    #converting labels into torch format
    train_labels = train_labels.astype(int);
    train_labels = torch.from_numpy(train_labels)

    #converting validation seqs
    val_seqs = val_seqs.reshape(val_seqs.shape[0], 1, val_seqs.shape[1], val_seqs.shape[2])
    val_seqs = torch.from_numpy(val_seqs)
    #converting validation labels
    val_labels = val_labels.astype(int);
    val_labels = torch.from_numpy(val_labels)

    #print(train_seqs.shape, train_labels.shape)
    #print(val_seqs.shape, val_labels.shape)

    #sys.exit()
    #defining a model
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        #torch.nn.Conv2d(42*4, 100, kernel_size=(2, 2)),
        torch.nn.Linear(42*4, 100),
        torch.nn.ReLU(),
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    #defining an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)

    #training networks
    x_train, y_train = Variable(train_seqs), Variable(train_labels)
    print(x.shape, y.shape)
    #sys.exit()
    x_val, y_val = Variable(val_seqs), Variable(val_labels)
    pred = model(x)

    #get loss
    loss = loss_fn(pred, y)

    #backpropagation
    loss.backward()
    optimizer.step()
    cost = loss.data[0]

    print(cost)


#trues = get_seqs(arg.true, arg.nt, arg.start, arg.end)
#fakes = get_seqs(arg.fake, arg.nf, arg.start, arg.end)


#acc = pytorch_classification(arg.true, arg.fake, arg.epoch, arg.batch)
'''
