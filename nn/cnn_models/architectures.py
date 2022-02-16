#!/usr/bin/python3

"""
## Class definitions for CNNs for GenDL ##
"""

import functools
from math import ceil
import operator
import sys

import torch
from torch.nn import Dropout, Linear, LazyLinear, Module, ModuleList, Conv2d
from torch.nn import MaxPool2d
from torch.nn import ELU as elu
from torch.nn import ReLU as relu
from torch.nn import Sigmoid as sigmoid
from torch.nn import BatchNorm2d, BatchNorm1d
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
import torch.nn as nn

from gendl.training_tools import layers_list


cnn_template = {
	'channels'      : [1],
	'conv_ks'       : [(1, 1)],
	'pool_ks'       : [(1, 1)]
}


class MLP(nn.Module):
	
	def __init__(self, input_dim, layers):
		super().__init__()
		self.hidden = ModuleList()
		self.act = ModuleList()
		
		prev = input_dim
		for i in range(1, len(layers)):
			input = layers[i-1]
			output = layers[i]
			self.hidden.append(Linear(input, output))
		
		for i in range(len(self.hidden) - 1):
			kaiming_uniform_(self.hidden[i].weight, nonlinearity='relu')
			self.act.append(nn.ReLU())
		xavier_uniform_(self.hidden[-1].weight)
		self.act.append(nn.Sigmoid())
	
	def forward(self, X):
		for hidden, act in zip(self.hidden, self.act):
			X = hidden(X)
			X = act(X)
		
		return X


class FlexFC(Module):
	def __init__(self, input_dim=None, layers=[256]):
		assert(input_dim is not None)
		
		super().__init__()
		self.net = []
		# self.net.append(
# 			Linear(
# 				in_features=input_dim,
# 				out_features=layers[0]
# 			)
# 		)
		
		prev = input_dim
		for i, nodes in enumerate(layers):
			self.net.append(
				Linear(
					in_features=prev,
					out_features=nodes
				)
			)
			#self.net.append(nn.BatchNorm1d(nodes))
			self.net.append(nn.ReLU())
			
			prev=nodes
		
		self.net.append(
			Linear(
				in_features=prev,
				out_features=1
			)
		)
		self.net.append(nn.Sigmoid())
		
		self.arch = nn.Sequential(*self.net)
	
	def forward(self, x):
		#batch_size = x.size(0)
		#x = x.view(batch_size, -1)
		#x = self.arch(x)
		#print(x.shape)
		#x = x.unsqueeze(0)
		#print(x.shape)
		return self.arch(x)
		
					
class SimpleCNN(Module):
	"""
	Class definition for a simple CNN working on one-hot encoded sequences
	All layers are convolutional.
	
	Parameters
	----------
	dropout: Dropout rate (optional)
		Optional dropout rate, between 0 and 1.
		float, 0 < dropout < 1
	
	Returns
	-------
	CNN model, PyTorch nn.Module object
	"""
	
	def __init__(self, dropout=None):

		if dropout is not None:
			assert(type(dropout) == float)
			assert(dropout < 1.0 and dropout > 0.0)
		
		super().__init__()
		
		# CNN layers
		self.conv1 = nn.Conv2d(
			in_channels=1,
			out_channels=80,
			kernel_size=(5,4),
			padding=(2,2)
		)
		
		self.conv2 = nn.Conv2d(
			in_channels=80,
			out_channels=50,
			kernel_size=(5,4),
			padding=(2,1)
		)
		self.relu = relu()
		self.fc1 = Linear(10200,250)
		self.fc2 = Linear(250, 10)
		self.fc3 = Linear(10, 1)
		
		self.pool1 = nn.MaxPool2d((3,3), stride=1, padding=1)
		self.pool2 = nn.MaxPool2d((5,5), stride=1, padding=2)
		
		self.dropout = Dropout(dropout) if dropout is not None else None
		
		self.sigmoid = nn.Sigmoid()
	
	def forward(self, features):
		# conv1
		#print('in', features.shape)
		x = self.conv1(features)
		#print('conv 1',x.shape)
		x = self.relu(x)
		x = self.pool1(x)
		#print('pool 1',x.shape)
		if self.dropout is not None: activate = self.dropout(x)
		
		# conv2
		x = self.conv2(x)
		#print('conv2', x.shape)
		x = self.relu(x)
		x = self.pool2(x)
		#print('pool2', x.shape)
		if self.dropout is not None: activate = self.dropout(x)
		# print(x.shape)
		# flatten and classify
		x = torch.flatten(x, 1)
		#print('flattened', x.shape)
		x = self.relu(self.fc1(x))
		x = self.relu(self.fc2(x))
		x = self.fc3(x)
		#print('classify', x.shape)
		return self.sigmoid(x)


class FlexCNN(nn.Module):
	"""
	Flexible CNN architecture definition for binary classification of genomic
	sequences
	"""
	
	def __init__(
		self,
		input_dim=None,
		layers={
			'channels':[1],
			'conv_ks':[(5,4)],
			'pool_ks':[(3,3)]
		},
		fc=[250],
		cnn_dropout=[0.1],
		fc_dropout=[0.1]
	):
		assert(input_dim is not None)
		
		if cnn_dropout is not None:
			assert(type(cnn_droout) == list)
			assert(len(cnn_dropout) == len(layers['channels']))
			for dr in cnn_dropout:
				assert(type(dr) == float)
				assert(dr < 1.0 and dr > 0.0)
		else: cnn_dropout = [None] * len(layers['channels'])
		
		if fc_dropout is not None:
			assert(type(fc_dropout) == list)
			assert(len(fc_dropout) == len(fc))
			for dr in fc_dropout:
				assert(type(dr) == float)
				assert(dr < 1.0 and dr > 0.0)
		else: fc_dropout = [None] * len(fc)
		
		super().__init__()
		
		layer_defs = layers_list(layers, cnn_template)
		
		cnn_layers = []
		prev=1
		for dr, layer in zip(cnn_dropout, layer_defs):
			cnn_layers.append(
				nn.Conv2d(
					in_channels=prev,
					out_channels=layer.channels,
					kernel_size=layer.conv_ks,
					stride=1,
					padding='same'
				)
			)
			pool_pad = (
				int((layer.pool_ks[0] - 1) / 2),
				int((layer.pool_ks[1] - 1) / 2)
			)
			cnn_layers.append(nn.ELU())
			cnn_layers.append(
				nn.MaxPool2d(
					layer.pool_ks,
					stride=1,
					padding=pool_pad
				)
			)
			if dr is not None: cnn_layers.append(Dropout(dr))
			prev = layer.channels
			
		self.cnn_features = nn.Sequential(*cnn_layers)
	
		cnn_num_features = functools.reduce(
			operator.mul,
			list(self.cnn_features(torch.rand(1, *input_dim)).shape)
		)
		
		classifier_layers = []
		prev = cnn_num_features
		for dr, nodes in zip(fc_dropout, fc):
			classifier_layers.append(
				Linear(
					in_features=prev,
					out_features=nodes
				)
			)
			classifier_layers.append(nn.ELU())
			if dr is not None: classifier_layers.append(Dropout(dr))
			prev = nodes
		
		classifier_layers.append(
			Linear(
				in_features=prev,
				out_features=1
			)
		)
		classifier_layers.append(nn.Sigmoid())
		
		self.classifier = nn.Sequential(*classifier_layers)
	
	def forward(self, x):
		batch_size = x.size(0)
		
		x = self.cnn_features(x)
		x = x.view(batch_size, -1)
		x = self.classifier(x)
		
		return x
		

class BatchNormCNN(nn.Module):
	"""
	Flexible CNN definitions with BatchNormalization
	"""
	def __init__(
		self,
		input_dim=None,
		layers={
			'channels':[1],
			'conv_ks':[(5,4)],
			'pool_ks':[(3,3)]
		},
		fc=[250]
	):
		assert(input_dim is not None)
		
		super().__init__()
		
		layer_defs = layers_list(layers, cnn_template)
		
		cnn_layers = []
		prev=1
		for layer in layer_defs:
			cnn_layers.append(
				nn.Conv2d(
					in_channels=prev,
					out_channels=layer.channels,
					kernel_size=layer.conv_ks,
					stride=1,
					padding='same'
				)
			)
			cnn_layers.append(BatchNorm2d(layer.channels))
			cnn_layers.append(nn.ELU())
			pool_pad = (
				int((layer.pool_ks[0] - 1) / 2),
				int((layer.pool_ks[1] - 1) / 2)
			)
			cnn_layers.append(
				nn.MaxPool2d(
					layer.pool_ks,
					stride=1,
					padding=pool_pad
				)
			)
			prev = layer.channels
			
		self.cnn_features = nn.Sequential(*cnn_layers)
	
		cnn_num_features = functools.reduce(
			operator.mul,
			list(self.cnn_features(torch.rand(1, *input_dim)).shape)
		)
		
		classifier_layers = []
		prev = cnn_num_features
		for nodes in fc:
			classifier_layers.append(
				Linear(
					in_features=prev,
					out_features=nodes
				)
			)
			classifier_layers.append(BatchNorm1d(nodes))
			classifier_layers.append(nn.ELU())
			prev = nodes
		
		classifier_layers.append(
			Linear(
				in_features=prev,
				out_features=1
			)
		)
		classifier_layers.append(nn.Sigmoid())
		
		self.classifier = nn.Sequential(*classifier_layers)
	
	def forward(self, x):
		batch_size = x.size(0)
		
		x = self.cnn_features(x)
		x = x.view(batch_size, -1)
		x = self.classifier(x)
		
		return x


class ResBlock(nn.Module):
	def __init__(self, module):
		super().__init__()
		self.module = module
	
	def forward(self, inputs):
		return self.module(inputs) + inputs


class ResCNN(nn.Module):
	"""
	"""
	
	def __init__(
		self,
		input_dim=None,
		layers={
			'channels':[1],
			'conv_ks':[(5,4)]
		},
		fc=[250]
	):
		assert(input_dim is not None)
		
		super().__init__()
		
		layer_defs = layers_list(layers, cnn_template)
		
		cnn_layers = []
		cnn_layers.append(
			nn.Conv2d(
				in_channels=1,
				out_channels=layer_defs[0].channels,
				kernel_size=layer_defs[0].conv_ks,
				padding='same'
			)
		)
		prev=layer_defs[0].channels
		for i, layer in enumerate(layer_defs[1:]):
			print(i, prev)
			if i == 0:
				cnn_layers.append(
					nn.Sequential(
						nn.Conv2d(
							in_channels=prev,
							out_channels=layer.channels,
							kernel_size=layer.conv_ks,
							padding='same'
						),
						nn.ReLU()
					)
				)
			else:
				cnn_layers.append(
					ResBlock(
						nn.Sequential(
							nn.Conv2d(
								in_channels=prev,
								out_channels=layer.channels,
								kernel_size=layer.conv_ks,
								padding='same'
							),
							nn.BatchNorm2d(layer.channels),
							nn.ReLU()
						)
					)
				)
			prev = layer.channels
		
		self.cnn_features = nn.Sequential(*cnn_layers)
		
		cnn_num_features = functools.reduce(
			operator.mul,
			list(self.cnn_features(torch.rand(1, *input_dim)).shape)
		)
		
		classifier_layers = []
		prev = cnn_num_features
		for nodes in fc:
			classifier_layers.append(
				Linear(
					in_features=prev,
					out_features=nodes
				)
			)
			classifier_layers.append(BatchNorm1d(nodes))
			classifier_layers.append(nn.ReLU())
			prev = nodes
		
		classifier_layers.append(
			Linear(
				in_features=prev,
				out_features=1
			)
		)
		classifier_layers.append(nn.Sigmoid())
		
		self.classifier = nn.Sequential(*classifier_layers)		
	
	def forward(self, x):
		batch_size = x.size(0)
		
		x = self.cnn_features(x)
		x = x.view(batch_size, -1)
		x = self.classifier(x)
		
		return x


if __name__ == '__main__':
	
	import argparse
	from math import floor
	import sys
	
	from gendl import seqio
	from gendl.training_tools import model_validator, fit_model
	import numpy as np
	import torch
	import torch.nn as nn
	import torch.utils.data as data_utils
	import torch.optim as optim
	from torchinfo import summary
	import torchvision
	
	parser = argparse.ArgumentParser(
		description='Test CNN on genomic sequences')
	parser.add_argument(
		'--file1', required=True, type=str,
		metavar='<path>', help='path to fasta file for true sequences')
	parser.add_argument(
		'--file0', required=True, type=str,
		metavar='<path>', help='path to fasta file for false sequences')
	parser.add_argument(
		'--start', required=False, type=int, metavar='<int>',
		help='starting position of sub sequence to train/test on')
	parser.add_argument(
		'--stop', required=False, type=int, metavar='<int>',
		help='stopping position of sub sequence to train/test on')
	parser.add_argument(
		'--num', required=False, type=int, metavar='<int>', default=-1,
		help='number of sequences to train on [-1]')
	parser.add_argument(
		'--batchsize', required=False, type=int,
		metavar='<int>', default=128, help='training batch size')
	parser.add_argument(
		'--epochs', required=False, type=int,
		metavar='<int>', default=5, help='num of epochs to run')
	parser.add_argument(
		'--lrate', required=False, type=float,
		metavar='<float>', default=1e-3, help='learing rate')
	parser.add_argument(
		'--dropout', required=False, type=float,
		metavar='<float>', default=None, help='dropout rate')
	parser.add_argument(
		'--split', required=False, type=float, metavar='<float>',
		default=0.30, help='validation split')
	parser.add_argument(
		'--seed', required=False, type=int, metavar='<int>',
		default=1, help='setting random seed')
	parser.add_argument(
		'--l2', required=False, type=float, metavar='<float>',
		default=1e-3, help='l2 regularization weight')
	
	arg = parser.parse_args()
	
	# make linear one hot sequences
	s1f = seqio.fasta2onehot(arg.file1, 1)
	s2f = seqio.fasta2onehot(arg.file0, 0)
	
	pos_seqs   = []
	pos_labels = []
	neg_seqs   = []
	neg_labels = []
	for ps in s1f:
		seq = ps[:-1]
		seq = list(seq)
		seq = np.array(seq, dtype=np.float64)
		pos_seqs.append(seq)
	
	for ns in s2f:
		seq = ns[:-1]
		seq = list(seq)
		seq = np.array(seq, dtype=np.float64)
		neg_seqs.append(seq)
	
	pos_seqs = np.array(pos_seqs)
	neg_seqs = np.array(neg_seqs)
	# find how large train/test will be given arg.split
	test_size = floor(len(s1f) * arg.split)
	
	s1_train = pos_seqs[test_size:]
	s1_test  = pos_seqs[:test_size]
	s2_train = neg_seqs[test_size:]
	s2_test  = neg_seqs[:test_size]
	
	# put all train/test sequences together
	train_sequences = np.concatenate((s1_train, s2_train), axis=0)
	test_sequences  = np.concatenate((s1_test, s2_test),   axis=0)
	
	# put all train/test labels together
	train_labels = np.ones(s1_train.shape[0])
	train_labels = np.concatenate(
		(train_labels, np.zeros(s2_train.shape[0])),
		axis=0
	)
	train_labels = train_labels.reshape(train_labels.shape[0], 1)
	
	test_labels = np.ones(s1_test.shape[0])
	test_labels = np.concatenate(
		(test_labels, np.zeros(s2_test.shape[0])),
		axis=0
	)
	test_labels = test_labels.reshape(test_labels.shape[0], 1)
	"""
	# read in sequences and make numpy arrays of one-hot sequences
	s1 = seqio.seq2features(
		seqs=arg.file1,
		num=arg.num,
		start=arg.start,
		stop=arg.stop,
		seed=arg.seed,
		label=1
	)
	#print(s1.shape)
	s1 = s1.reshape(s1.shape[0], 1, s1.shape[1], s1.shape[2])
	#print(s1.shape)
	s2 = seqio.seq2features(
		seqs=arg.file0,
		num=arg.num,
		start=arg.start,
		stop=arg.stop,
		seed=arg.seed,
		label=0
	)
	s2 = s2.reshape(s2.shape[0], 1, s2.shape[1], s2.shape[2])
	
	# find how large train/test will be given arg.split
	test_size = floor(len(s1) * arg.split)
	
	s1_train = s1[test_size:]
	s1_test  = s1[:test_size]
	s2_train = s2[test_size:]
	s2_test  = s2[:test_size]
	
	rows = s1_train.shape[2]
	cols = s1_train.shape[3]
	
	# put all train/test sequences together
	train_sequences = np.concatenate((s1_train, s2_train), axis=0)
	test_sequences  = np.concatenate((s1_test, s2_test),   axis=0)
	
	#print(train_sequences[:25])
	#sys.exit()
	# put all train/test labels together
	train_labels = np.ones(s1_train.shape[0])
	train_labels = np.concatenate(
		(train_labels, np.zeros(s2_train.shape[0])),
		axis=0
	)
	train_labels = train_labels.reshape(train_labels.shape[0], 1)
	
	test_labels = np.ones(s1_test.shape[0])
	test_labels = np.concatenate(
		(test_labels, np.zeros(s2_test.shape[0])),
		axis=0
	)
	test_labels = test_labels.reshape(test_labels.shape[0], 1)
	"""
	# setting seed to just compare the results
	seed = arg.seed
	# setting the random seed from pytorch random number generators
	torch.manual_seed(seed)
	# enabling benchmark mode in cudnn (GPU accelerated library of primitives
	# for deep neural net)
	torch.backends.cudnn.benchmark = False
	# making experiments reproducible
	torch.backends.cudnn.deterministic = True
	
	# use gpu if available
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	fc_net = FlexFC(input_dim=len(s1f[0])-1, layers=[32,32,32,32,32])
	
	print(fc_net)
	
	"""
	# create a model from `SimpleCNN` class
	# load it to the specified device, either gpu or cpu
	cnn = SimpleCNN(dropout=arg.dropout).to(device)
	s_cnn = summary(
		cnn, input_size=(arg.batchsize, 1, rows, cols), verbose=0)
	cnn_str = repr(s_cnn)
	print(cnn_str.encode('utf-8').decode('latin-1'))
	print()
	"""
	# Set the data loaders
	
	train = data_utils.TensorDataset(
		torch.Tensor(train_sequences),
		torch.Tensor(train_labels)
	)
	
	train_loader = data_utils.DataLoader(
		train,
		batch_size=arg.batchsize,
		shuffle=True
	)
	
	test = data_utils.TensorDataset(
		torch.Tensor(test_sequences),
		torch.Tensor(test_labels)
	)
	
	test_loader = data_utils.DataLoader(
		test,
		batch_size=1,
		shuffle=True
	)
	
	criterion = nn.BCELoss()
	optimizer = optim.Adam(
		fc_net.parameters(),
		lr=arg.lrate,
		weight_decay=arg.l2
	)
	
	fc_net = fit_model(
		fc_net,
		train=train_loader,
		test=test_loader,
		optimizer=optimizer,
		criterion=criterion,
		device=device,
		epochs=arg.epochs
	)	
	sys.exit()
	"""
	# Set loss and optimizer
	criterion = nn.BCELoss()
	optimizer = optim.Adam(
		cnn.parameters(),
		lr=arg.lrate
	)
		#weight_decay=arg.l2)
	# Fit the model
	model_cnn = fit_model(
		cnn,
		train=train_loader,
		test=test_loader,
		optimizer=optimizer,
		criterion=criterion,
		device=device,
		epochs=arg.epochs
	)	
	# Make the flexible CNN
	flex_cnn = FlexCNN(
		layers={
			'channels':[128, 64, 64, 16],
			'conv_ks':[(6,4), (5,4), (4,4), (4,4)],
			'pool_ks':[(3,3), (3,3), (3,3), (3,3)]
		},
		fc=[256, 64, 10],
		cnn_dropout=None,
		fc_dropout=[0.5,0.5,0.5],
		input_dim=(1,51,4)
	)
	f_cnn = summary(
		flex_cnn, input_size=(arg.batchsize, 1, rows, cols), verbose=0)
	cnn_str = repr(f_cnn)
	print(cnn_str.encode('utf-8').decode('latin-1'))
	print()
	
	criterion = nn.BCELoss()
	optimizer = optim.Adam(
		flex_cnn.parameters(),
		lr=arg.lrate
	)
	
	flex_cnn = fit_model(
		flex_cnn,
		train=train_loader,
		test=test_loader,
		optimizer=optimizer,
		criterion=criterion,
		device=device,
		epochs=arg.epochs
	)
	
	# Make the flexible batchnormed CNN
	norm_cnn = BatchNormCNN(
		layers={
			'channels':[64, 32, 16],
			'conv_ks':[(6,4), (5,4), (4,4)],
			'pool_ks':[(3,3), (3,3), (3,3)]
		},
		fc=[256, 64],
		input_dim=(1,51,4)
	)
	b_cnn = summary(
		norm_cnn, input_size=(arg.batchsize, 1, rows, cols), verbose=0)
	cnn_str = repr(b_cnn)
	print(cnn_str.encode('utf-8').decode('latin-1'))
	print()
	
	criterion = nn.BCELoss()
	optimizer = optim.Adam(
		norm_cnn.parameters(),
		lr=arg.lrate
	)
	
	norm_cnn = fit_model(
		norm_cnn,
		train=train_loader,
		test=test_loader,
		optimizer=optimizer,
		criterion=criterion,
		device=device,
		epochs=arg.epochs
	)
	"""
	res_cnn = ResCNN(
		layers={
			'channels': [256, 80, 80, 80, 80, 80],
			'conv_ks':  [(6,1), (4,2), (3,3), (5,4), (5,4), (5,4)]
		},
		fc=[80, 16],
		input_dim=(1,51,4)
	)
	
	r_cnn = summary(
		res_cnn, input_size=(arg.batchsize, 1, rows, cols), verbose=0)
	cnn_str = repr(r_cnn)
	print(cnn_str.encode('utf-8').decode('latin-1'))
	print()
	
	criterion = nn.BCELoss()
	optimizer = optim.Adam(
		res_cnn.parameters(),
		lr=arg.lrate,
		weight_decay=arg.l2
	)
	
	res_cnn = fit_model(
		res_cnn,
		train=train_loader,
		test=test_loader,
		optimizer=optimizer,
		criterion=criterion,
		device=device,
		epochs=arg.epochs
	)	