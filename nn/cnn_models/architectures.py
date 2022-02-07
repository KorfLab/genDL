#!/usr/bin/python3

"""
## Class definitions for CNNs for GenDL ##
"""
import sys

from torch import relu, sigmoid
from torch.nn import Dropout, Linear, LazyLinear, Module, ModuleList, Conv2d
from torch.nn import MaxPool2d
from torch.nn import ConvTranspose2d
import torch.nn as nn
from training_tools import layers_list, cnn_ae_validator
from training_tools import encoder_template, decoder_template

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
			out_channels=100,
			kernel_size=4,
			padding=1,
			stride=1)
		
		self.conv2 = nn.Conv2d(
			in_channels=100,
			out_channels=100,
			kernel_size=4,
			padding=1,
			stride=1)
		
		self.classify = LazyLinear(1)
		
		self.pool = nn.MaxPool2d(4, stride=1, padding=1)
		
		self.dropout = Dropout(dropout) if dropout is not None else None
	
	def forward(self, features):
		# conv1
		# print('in', features.shape)
		x = self.conv1(features)
		# print('conv 1',x.shape)
		x = relu(x)
		x = self.pool(x)
		# print('pool 1',x.shape)
		if self.dropout is not None: activate = self.dropout(x)
		
		# conv2
		x = self.conv2(x)
		# print(x.shape)
		x = relu(x)
		x = self.pool(x)
		# print(x.shape)
		if self.dropout is not None: activate = self.dropout(x)
		
		# flatten and classify
		x = torch.flatten(x, 1)
		x = self.classify(x)
		
		return x.sigmoid()

if __name__ == '__main__':
	
	import argparse
	from math import floor
	import sys
	
	from gendl import seqio
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
		'--num', required=False, type=int, metavar='<int>', default=-1
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
	
	arg = parser.parse_args()
	
	# read in sequences and make numpy arrays of one-hot sequences
	s1 = seqio.seq2features(
		seqs=arg.file1,
		num=arg.num,
		start=arg.start,
		stop=arg.stop,
		seed=arg.seed,
		label=1
	)
	
	s2 = seqio.seq2features(
		seqs=arg.file0,
		num=arg.num,
		start=arg.start,
		stop=arg.stop,
		seed=arg.seed,
		label=0
	)
	
	# find how large train/test will be given arg.split
	test_size = floor(len(s1) * arg.split)
	
	s1_train = s1[test_size:]
	s1_test  = s1[:test_size]
	s2_train = s2[test_size:]
	s2_test  = s2[:test_size]
	
	rows = s1_train[0].shape[0]
	cols = s1_train[0].shape[1]
	
	# put all train/test sequences together
	train_sequences = np.concatenate((s1_train, s2_train), axis=0)
	test_sequences  = np.concatenate((s1_test, s2_test),   axis=0)
	
	# put all train/test labels together
	train_labels = np.ones(s1_train.shape[0])
	np.concatenate(np.zeros(s2_train.shape[0], axis=0)
	
	test_labels = np.ones(s1_test.shape[0])
	np.concatenate(np.zeros(s1_test.shape[0]))
	
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
	
	# create a model from `SimpleCNN` class
	# load it to the specified device, either gpu or cpu
	
	cnn = SimpleCNN(dropout=arg.dropout).to(device)
	s_cnn = summary(
		cnn, input_size=(arg.batchsize, 1, rows, cols), verbose=0)
	su_aefc = repr(s_aefc)
	print(su_aefc.encode('utf-8').decode('latin-1'))
	print()
	
	# Set the data loaders
	
	train = data_utils.TensorDataset(
		torch.Tensor(train_sequences),
		torch.Tensor(train_labels)
	)
	
	train_loader = data_utils.DataLoader(
		train,
		batch_size=arg.batchsize,
		shuffle=True)
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
	
	# Set loss and optimizer
	criterion = nn.BCELoss()
	optimizer = optim.Adam(
		cnn.parameters(),
		lr=arg.lrate
	)
		#weight_decay=arg.l2)
	
	# Fit the model
	model_cnn = fit_model(
		model_aefc,
		train=train_loader,
		test=test_loader,
		optimizer=optimizer,
		criterion=criterion,
		device=device,
		epochs=arg.epochs)

	# visualization
	if arg.vis:
		# initial
		before = pdb_writer(
			coords=df.xyz_set[trn], seq=df.fragment_seq[trn],
			atoms=[df.fragment_type[trn][0]] * len(df.fragment_seq[trn]),
			chain=[df.chain_id[trn][0]] * len(df.fragment_seq[trn]))
		# normalized
		norm = pdb_writer(
			coords=[df.norm_frag[trn][i:i + 3] for i in range(
				0, len(df.norm_frag[trn]), 3)],
			seq=df.fragment_seq[trn],
			atoms=[df.fragment_type[trn][0]] * len(df.fragment_seq[trn]),
			chain=[df.chain_id[trn][0]] * len(df.fragment_seq[trn]))
		# normalized after training
		saved = model_aefc.forward(test[0][0])
		reconstructed = pdb_writer(
			coords=[saved[i:i + 3] for i in range(0, len(saved), 3)],
			seq=df.fragment_seq[trn],
			atoms=[df.fragment_type[trn][0]] * len(df.fragment_seq[trn]),
			chain=[df.chain_id[trn][0]] * len(df.fragment_seq[trn]))
		
		with open('./images/before.pdb', 'w') as fp:
			fp.write(before)
		fp.close()
		with open('./images/norm.pdb', 'w') as fp:
			fp.write(norm)
		fp.close()
		with open('./images/reconstructed.pdb', 'w') as fp:
			fp.write(reconstructed)
		fp.close()
	
	model_dyn_aefc = DynamicAEfc(
		inshape=fshape,
		dropouts=[0.25] * 7,
		units=[256, 128, 64, 16, 64, 128, 256],
		function_list=[relu] * 7).to(device)
	
	s_dyn_aefc = summary(
		model_dyn_aefc,
		input_size=(arg.batchsize, 1, fshape),
		verbose=0)
	su_dyn_aefc = repr(s_dyn_aefc)
	print(su_dyn_aefc.encode('utf-8').decode('latin-1'))
	print()
	
	# Set optimizer
	optimizer = optim.Adam(
		model_dyn_aefc.parameters(),
		lr=arg.lrate,
		weight_decay=1e-6)
	
	# Fit the model
	model_dyn_aefc = fit_model(
		model_dyn_aefc,
		train=train_loader,
		test=test_loader,
		optimizer=optimizer,
		criterion=criterion,
		epochs=arg.epochs,
		device=device)
	
	model_aecnn = SimpleAEcnn(dropout=arg.dropout).to(device)
	
	s_aecnn = summary(
		model_aecnn,
		input_size=(arg.batchsize, 1, dshape, dshape),
		verbose=0)
	
	su_aecnn = repr(s_aecnn)
	print(su_aecnn.encode('utf-8').decode('latin-1'))
	print()
	
	# Set optimizer
	optimizer = optim.Adam(
		model_aecnn.parameters(),
		lr=arg.lrate,
		weight_decay=arg.l2)
	
	# Set the data loaders
	train_coords = np.array(df.dmatrix[:trn].to_list())
	test_coords  = np.array(df.dmatrix[trn:].to_list())
	train = data_utils.TensorDataset(
		torch.Tensor(train_coords),
		torch.Tensor(train_coords))
	
	train_loader = data_utils.DataLoader(
		train,
		batch_size=arg.batchsize,
		shuffle=True)
	
	test = data_utils.TensorDataset(
		torch.Tensor(test_coords),
		torch.Tensor(test_coords))
	
	test_loader = data_utils.DataLoader(test, batch_size=1, shuffle=True)
	
	
	# Fit the model
	model_aecnn = fit_model(
		model_aecnn,
		train=train_loader,
		test=test_loader,
		optimizer=optimizer,
		criterion=criterion,
		epochs=10,
		device=device)
	
	# Set the dynamic cnn
	encoder={
		'channels': [100,50],
		'conv_ks': [(4,4), (4,4)],
		'pool_ks': [(4,4), (4,4)],
		'conv_paddings': [(1,1), (1,1)],
		'pool_paddings': [(1,1), (1,1)]}
	decoder={
		'channels': [100,1],
		'convt_ks': [(3,3), (3,3)],
		'convt_strides': [(1,1), (1,1)]}
	
	if not cnn_ae_validator(
		inshape=(dshape, dshape), 
		encoder=encoder,
		decoder=decoder):
		raise('cnn validator failed')
	
	
	model_dyn_cnn = DynamicAEcnn(
		channels=[100, 50, 100, 1],
		dropout=None,
		encoder={
			'channels': [100,50],
			'conv_ks': [(4,4), (4,4)],
			'pool_ks': [(4,4), (4,4)],
			'conv_paddings': [(1,1), (1,1)],
			'pool_paddings': [(1,1), (1,1)]},
		decoder={
			'channels': [100,1],
			'convt_ks': [(3,3), (3,3)],
			'convt_strides': [(1,1), (1,1)]})
	
	optimizer = optim.Adam(
		model_dyn_cnn.parameters(),
		lr=arg.lrate,
		weight_decay=arg.l2)
	
	d_aecnn = summary(
		model_dyn_cnn,
		input_size=(arg.batchsize, 1, dshape, dshape),
		verbose=0)
	
	dy_aecnn = repr(d_aecnn)
	print(dy_aecnn.encode('utf-8').decode('latin-1'))
	print()
	
	# Fit the model
	model_dyn_cnn = fit_model(
		model_dyn_cnn,
		train=train_loader,
		test=test_loader,
		optimizer=optimizer,
		criterion=criterion,
		epochs=10,
		device=device)
	print()