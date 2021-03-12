#binary flat data
import argparse
import sys
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable

from sklearn.model_selection import train_test_split

from gendl import pwm, seqio

def conv(seqs):
	df = []
	for seq in seqs:
		split_seq = [float(i) for i in seq]
		df.append(split_seq)
	df = pd.DataFrame(df)
	return (df)


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
	parser.add_argument('--batch', required=False, type=int, default = 1,
		metavar='<int>', help='batch size')
	parser.add_argument('--seed', required=False, type=int,
		metavar='<int>', help='random seed')
	arg = parser.parse_args()

	assert(0<arg.split<1.0)
	assert(type(arg.epoch) == int)
	assert(type(arg.batch) == int)

	if arg.seed:
		random.seed(arg.seed)

	seqs1 = seqio.fasta2binary(arg.file1, '1')
	seqs0 = seqio.fasta2binary(arg.file0, '0')
	seqs = seqs1 + seqs0
	random.shuffle(seqs)
	#print(seqs[0], len(seqs[0]))

	#reformatting data and extracting labels and data
	conv_df = conv(seqs)

	y = conv_df.iloc[:, -1].values
	X = conv_df.iloc[:, :-1].values


	#splitting data into train and test sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = arg.split)
	print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

	#creating tensor for train data
	torch_X_train = torch.from_numpy(X_train).type(torch.LongTensor)
	torch_y_train = torch.from_numpy(y_train).type(torch.LongTensor)
	print(torch_X_train.shape, torch_y_train.shape)

	#creating tensor for test data
	torch_X_test = torch.from_numpy(X_test).type(torch.LongTensor)
	torch_y_test = torch.from_numpy(y_test).type(torch.LongTensor)
	print(torch_X_test.shape, torch_y_test.shape)

	#reshaping data

	#[batch, channels, height, width]
	#since it is already in 2d shape = no need to reshape it

	torch_X_train = torch_X_train.view(X_train.shape[0], 1, X_train.shape[1]).float()
	torch_X_test = torch_X_test.view(X_test.shape[0], 1, X_test.shape[1]).float()

	#print(torch_X_train.shape, torch_X_train)

	#creates a tensorDataSet from a vector of tensors
	train = torch.utils.data.TensorDataset(torch_X_train,torch_y_train)
	test = torch.utils.data.TensorDataset(torch_X_test,torch_y_test)

	#data loader
	train_loader = torch.utils.data.DataLoader(train, batch_size = arg.batch, shuffle = False)
	test_loader = torch.utils.data.DataLoader(test, batch_size = arg.batch, shuffle = False)


	class CNN(nn.Module):
		def __init__(self):
			super(CNN, self).__init__()
			#convolutional layer sees (height, width, depth) = (1, 84, 2)
			self.conv1 = nn.Conv1d(in_channels=1, out_channels = 1, kernel_size = 3)

			#len - kernel_size + 1
			#if it is will it jump off
			#in_channels = 2 for binary and 4 for onehot

			#after maxpool = (1, 42, 2)
			self.fc1 = nn.Linear(41, 1)

			"""
			note for the future binary
			super(CNN, self).__init__()
			#input will be restructed
			the overall = (16000, 42, 2)
			#input shape = (h, w, d) = (1, 42, 2)
			"""

		def forward(self, x):
			x = F.relu(self.conv1(x))
			x = F.relu(F.max_pool1d(x, 2))
			#print(x.size())
			x = x.view(-1, 41)
			#print(x.size())
			x = F.relu(self.fc1(x))
			x = torch.sigmoid(x)
			#print(x.size(), x)
			'''

			x = F.relu(F.max_pool1d(self.conv1(x), 2))
			x = x.view(-1, 1*1*41)
			x = F.relu(self.fc1(x))
			x = F.sigmoid(x)
			'''

			return x

	model = CNN()
	print(model)

	def fit (model, train_loader, epochs):
		optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
		criterion = nn.BCELoss()

		for epoch in range(1, epochs):
			train_loss = 0.0
			correct_predicted = 0

			model.train()

			for data, target in train_loader:
				#clearing the gradient of all optimized variables
				optimizer.zero_grad()
				#forward_pass
				output = model(data)
				#calculating batch loss
				target = target.unsqueeze(1)
				target = target.float()
				#print(target, output.data)
				#print(output, target)
				#sys.exit()

				loss = criterion(output, target)
				#backward pass
				loss.backward()
				#optimization step
				optimizer.step()
				predicted = torch.max(output.data, 1)[1]
				print(predicted)
				sys.exit()
				correct_predicted += (predicted == target).sum()
				train_loss += loss.item()*data.size(0)



			#validation
			train_loss = train_loss/len(train_loader.sampler)
			correct_predicted = correct_predicted/len(train_loader.sampler)
			print(f'Epoch: {epoch} \tTraining loss: {train_loss:.6f} \t#Predicted: {correct_predicted:.2f}')

	fit(model, train_loader, arg.epoch)







