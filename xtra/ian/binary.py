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
from torch.nn import Module
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
		return random_split(self, (train_size, test_size)) # originally list

class MLP(Module):

	def __init__(self, n_inputs, l1, l2):
		super(MLP, self).__init__()
		
		self.hidden1 = Linear(n_inputs, l1)
		kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
		self.act1 = ReLU()
		
		self.hidden2 = Linear(l1, l2)
		kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
		self.act2 = ReLU()
		
		self.hidden3 = Linear(l2, 1)
		xavier_uniform_(self.hidden3.weight)
		self.act3 = Sigmoid()

	def forward(self, X):
		X = self.hidden1(X)
		X = self.act1(X)
		
		X = self.hidden2(X)
		X = self.act2(X)
		
		X = self.hidden3(X)
		X = self.act3(X)
		
		return X

def prepare_data(path, split):
	dataset = CSVDataset(path)
	size = len(dataset[0][0])
	train, test = dataset.get_splits(split)
	train_dl = DataLoader(train, batch_size=32, shuffle=True)
	test_dl = DataLoader(test, batch_size=1024, shuffle=False) # batch?
	return size, train_dl, test_dl

def train_model(train_dl, model):
	criterion = BCELoss() # or CrossEntropyLoss, MSELoss
	optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9) # or Adam

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

def fasta2binary(file, label):
	data = []
	for name, seq in seqio.read_fasta(file):
		s = ''
		for nt in seq:
			if   nt == 'A': s += '00'
			elif nt == 'C': s += '01'
			elif nt == 'G': s += '10'
			elif nt == 'T': s += '11'
			else: raise()
		s += str(label)
		data.append(s)
	return data

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--file1', required=True, type=str,
		metavar='<file>', help='fasta file of observed')
	parser.add_argument('--file0', required=True, type=str,
		metavar='<file>', help='fasta file of not observed')
	parser.add_argument('--layer2', required=False, type=int, default=168,
		metavar='<int>', help='layer 2 [%(default)i]')
	parser.add_argument('--layer3', required=False, type=int, default=168,
		metavar='<int>', help='layer 3 [%(default)i]')
	parser.add_argument('--iter', required=False, type=int, default=4,
		metavar='<int>', help='number of times to run [%(default)i]')
	parser.add_argument('--seed', required=False, type=int,
		help='set random seed')
	arg = parser.parse_args()
	
	if arg.seed: torch.manual_seed(1)
	
	## read fasta files and convert to a single binary encoded csv
	s1 = fasta2binary(arg.file1, 1)
	s2 = fasta2binary(arg.file0, 0)
	seqs = s1 + s2
	random.shuffle(seqs)
	
	csv = 'temp.csv'
	with open(csv, 'w') as fp:
		for item in seqs:
			fp.write(','.join(item))
			fp.write('\n')
	
	## do the deep learning stuff
	accs = []
	for x in range(arg.iter):
		size, train_dl, test_dl = prepare_data(csv, 0.5)
		model = MLP(size, arg.layer2, arg.layer3) # define the network
		train_model(train_dl, model) # train the model
		acc, f1 = evaluate_model(test_dl, model) # evaluate the model
		sys.stderr.write(f'{x} {size} {arg.layer2} {arg.layer3} {acc:.3f}\n')
		accs.append(acc)
	
	## finish up
	print(arg.file1, arg.file0, 168, arg.layer2, arg.layer3, statistics.mean(accs))
	os.remove(csv)
	
	
	
