import argparse
import io
import os
import random
import statistics
import sys

from numpy import vstack
from pandas import read_csv

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module, ModuleList
from torch.optim import SGD
from torch.nn import BCELoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_

from gendl import seqio

class CSVDataset(Dataset):

	def __init__(self, path):
		df = read_csv(path, header=None) # load the csv file as a dataframe
		self.X = df.values[:, :-1] # store the inputs 
		self.y = df.values[:, -1]  # and outputs
		self.X = self.X.astype('float32') # ensure input data is floats
		self.y = LabelEncoder().fit_transform(self.y) # label target 
		self.y = self.y.astype('float32') # ensure floats
		self.y = self.y.reshape((len(self.y), 1))

	def __len__(self):
		return len(self.X)

	def __getitem__(self, idx):
		return [self.X[idx], self.y[idx]]

	def get_splits(self, n_test):
		test_size = round(n_test * len(self.X))
		train_size = len(self.X) - test_size
		return random_split(self, (train_size, test_size))

class MLP(Module):

	def __init__(self, n_inputs, layers):
		super(MLP, self).__init__()
		
		self.hidden = ModuleList()
		self.act = ModuleList()
		for i in range(1, len(layers)):
			input = layers[i-1]
			output = layers[i]
			self.hidden.append(Linear(input, output))
		
		for i in range(len(self.hidden) -1):
			kaiming_uniform_(self.hidden[i].weight, nonlinearity='relu')
			self.act.append(ReLU())
		xavier_uniform_(self.hidden[-1].weight)
		self.act.append(Sigmoid())

	def forward(self, X):
		for hidden, act in zip(self.hidden, self.act):
			X = hidden(X)
			X = act(X)
		return X

def prepare_data(path, split):
	dataset = CSVDataset(path)
	train, test = dataset.get_splits(split)
	# why are the batch sizes hard-coded?
	train_dl = DataLoader(train, batch_size=32, shuffle=True)
	test_dl = DataLoader(test, batch_size=1024, shuffle=False)
	return train_dl, test_dl

def train_model(train_dl, model, r, m):
	criterion = BCELoss() # or CrossEntropyLoss, MSELoss
	optimizer = SGD(model.parameters(), lr=r, momentum=m) # or Adam
	for epoch in range(100):
		for i, (inputs, targets) in enumerate(train_dl): # mini batches
			optimizer.zero_grad() # clear the gradients
			yhat = model(inputs) # compute the model output
			loss = criterion(yhat, targets) # calculate loss
			loss.backward() # credit assignment
			optimizer.step() # update model weights

def evaluate_model(test_dl, model):
	predictions, actuals = list(), list()
	for i, (inputs, targets) in enumerate(test_dl):
		yhat = model(inputs) # evaluate the model on the test set
		yhat = yhat.detach().numpy() # retrieve numpy array
		actual = targets.numpy()
		actual = actual.reshape((len(actual), 1))
		yhat = yhat.round() # round to class values
		predictions.append(yhat) # store
		actuals.append(actual)
	
	predictions, actuals = vstack(predictions), vstack(actuals)
	acc = accuracy_score(actuals, predictions)
	f1 = f1_score(actuals, predictions, average='weighted')
	return acc, f1

def predict(row, model):
	row = Tensor([row]) # convert row to data (can be tuple)
	yhat = model(row) # make prediction
	yhat = yhat.detach().numpy() # retrieve numpy array
	return yhat

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--file1', required=True, type=str,
		metavar='<file>', help='fasta file of observed')
	parser.add_argument('--file0', required=True, type=str,
		metavar='<file>', help='fasta file of not observed')
	parser.add_argument('--layers', required=True, type=int, nargs='*',
		metavar='<int>', help='nodes in each hidden layer, e.g. 168 42 21 1')
	parser.add_argument('--rate', required=False, type=float, default=0.01,
		metavar='<float>', help='learning rate [%(default)f]')
	parser.add_argument('--momentum', required=False, type=float, default=0.9,
		metavar='<float>', help='momentum [%(default)f]')
	parser.add_argument('--iter', required=False, type=int, default=4,
		metavar='<int>', help='number of times to run [%(default)i]')
	parser.add_argument('--seed', required=False, type=int,
		help='set random seed')
	arg = parser.parse_args()
	
	if arg.seed: torch.manual_seed(1)
	
	# read fasta files and convert to a single one-hot encoded csv
	s1 = seqio.fasta2onehot(arg.file1, 1)
	s2 = seqio.fasta2onehot(arg.file0, 0)
	seqs = s1 + s2
	random.shuffle(seqs)
	csv = 'temp.csv'
	with open(csv, 'w') as fp:
		for item in seqs:
			fp.write(','.join(item))
			fp.write('\n')
	size = len(s1[0]) -1 # length of sequence
	
	# check network architecture
	if len(arg.layers) < 2: raise Exception('need at least 2 layers')
	if arg.layers[0] != size: raise Exception('input layer != inputs')
	if arg.layers[-1] != 1: raise Exception('last layer must be 1')
	
	# do the deep learning stuff
	accs = []
	for x in range(arg.iter):
		train_dl, test_dl = prepare_data(csv, 0.5)
		model = MLP(size, arg.layers)
		train_model(train_dl, model, arg.rate, arg.momentum)
		acc, f1 = evaluate_model(test_dl, model)
		sys.stderr.write(f'{x} {size} {arg.layers} {acc:.3f}\n')
		accs.append(acc)
	
	# finish up
	arch = [size] + arg.layers + [0]
	print(arg.file1, arg.file0, arch, statistics.mean(accs))
	os.remove(csv)
	