import argparse
import sys
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable

from sklearn.model_selection import train_test_split

from gendl import pwm, seqio

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='CNN model')
	parser.add_argument('--file1', required=True, type=str,
		metavar='<path>', help='path to fasta file1')
	parser.add_argument('--file0', required=True, type=str,
		metavar='<path>', help='path to fasta file0')
	parser.add_argument('--split', required=False, type=float, default = 0.2,
		metavar = '<float>', help = 'split size')
	parser.add_argument('--epoch', required=False, type=int, default = 10,
		metavar='<int>', help='number of epochs')
	parser.add_argument('--batch', required=False, type=int, default = 2,
		metavar='<int>', help='batch size')
	parser.add_argument('--seed', required=False, type=int,
		metavar='<int>', help='random seed')
	arg = parser.parse_args()

	assert(0<arg.split<1.0)
	assert(type(arg.epoch) == int)
	assert(type(arg.batch) == int)

	if arg.seed:
		random.seed(arg.seed)

	seqs1 = ['1' + seq for name, seq in seqio.read_fasta(arg.file1)]
	seqs0 = ['0' + seq for name, seq in seqio.read_fasta(arg.file0)]
	seqs = seqs1 + seqs0
	random.shuffle(seqs)

	#converting data to df

	conv_df = seqio.conv_data(seqs)
	#print(conv_df.shape)
	y = conv_df.iloc[:, 0].values
	X = conv_df.iloc[:, 1:].values

	#splitting data into train and test sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = arg.split)
	print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

	#creating tensor for train data
	#print('initial', X_train.shape)
	torch_X_train = torch.from_numpy(X_train).type(torch.LongTensor)
	torch_y_train = torch.from_numpy(y_train).type(torch.LongTensor)
	#print(torch_X_train.shape)

	#creating tensor for test data
	torch_X_test = torch.from_numpy(X_test).type(torch.LongTensor)
	torch_y_test = torch.from_numpy(y_test).type(torch.LongTensor)

	#reshape data
	#ask if needs to be reshaped or not
	torch_X_train = torch_X_train.view(-1, 1,1,42).float()
	torch_X_test = torch_X_test.view(-1,1,1,42).float()
	#print(torch_X_train.shape)
	#print(torch_X_test.shape)

	#creates a tensorDataSet from a vector of tensors
	train = torch.utils.data.TensorDataset(torch_X_train,torch_y_train)
	test = torch.utils.data.TensorDataset(torch_X_test,torch_y_test)

	#data loader
	train_loader = torch.utils.data.DataLoader(train, batch_size = arg.batch, shuffle = False)
	test_loader = torch.utils.data.DataLoader(test, batch_size = arg.batch, shuffle = False)

	class CNN(nn.Module):
		def __init__(self):
			super(CNN, self).__init__()
			self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
			self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
			self.conv3 = nn.Conv2d(32,64, kernel_size=5)
			self.fc1 = nn.Linear(3*3*64, 256)
			self.fc2 = nn.Linear(256, 10)

		def forward(self, x):
			x = F.relu(self.conv1(x))
			#x = F.dropout(x, p=0.5, training=self.training)
			x = F.relu(F.max_pool2d(self.conv2(x), 2))
			x = F.dropout(x, p=0.5, training=self.training)
			x = F.relu(F.max_pool2d(self.conv3(x),2))
			x = F.dropout(x, p=0.5, training=self.training)
			x = x.view(-1,3*3*64 )
			x = F.relu(self.fc1(x))
			x = F.dropout(x, training=self.training)
			x = self.fc2(x)
			return F.log_softmax(x, dim=1)


cnn = CNN()
print(cnn)

it = iter(train_loader)
X_batch, y_batch = next(it)
print(cnn.forward(X_batch).shape)



