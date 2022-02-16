#!/usr/bin/env python

import argparse
from math import floor
import random
import sys

from architectures import FlexFC, FlexCNN, BatchNormCNN, ResCNN, MLP
from gendl import seqio
from gendl.training_tools import model_validator, fit_model, make_1d_loaders
from gendl.training_tools import make_2d_loaders
import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
import torch.utils.data as data_utils
import torch.optim as optim
from torchinfo import summary
import torchvision

parser = argparse.ArgumentParser(
	description='Test neural networks on genomic sequences')
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
	default=0.25, help='validation split')
parser.add_argument(
	'--seed', required=False, type=int, metavar='<int>',
	default=None, help='setting random seed')
parser.add_argument(
	'--l2', required=False, type=float, metavar='<float>',
	default=1e-3, help='l2 regularization weight')
parser.add_argument(
	'--modelsave', required=True, type=str, metavar='<path>',
	help='name of file to save model to')
parser.add_argument(
	'--results', required=True, type=str, metavar='<path>',
	help='name of file to save accuracy results to')

arg = parser.parse_args()

"""
train_loader, test_loader = make_1d_loaders(
	file1=arg.file1,
	file0=arg.file0,
	num=arg.num,
	start=arg.start,
	stop=arg.stop,
	split=arg.split,
	batch=arg.batchsize,
	seed=arg.seed
)
"""
# change the loaders if using FC or CNN

train_loader, test_loader = make_2d_loaders(
	file1=arg.file1,
	file0=arg.file0,
	num=arg.num,
	start=arg.start,
	stop=arg.stop,
	split=arg.split,
	batch=arg.batchsize,
	seed=arg.seed	
)

# setting seed to just compare the results
if arg.seed != None:
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

# describe model here
"""
model = MLP(
	input_dim=4*(arg.stop-arg.start),
	layers=[4*(arg.stop-arg.start)+1, 1]
)
"""

model = ResCNN(
	layers={
		'channels': [256, 32, 32, 32, 32, 32],
		'conv_ks':  [(6,1), (4,2), (3,3), (5,4), (5,4), (5,4)]
	},
	fc=[50,10],
	input_dim=(1,arg.stop-arg.start,4)
)


print(model)
#sys.exit()
criterion = nn.BCELoss()
optimizer = optim.Adam(
	model.parameters(),
	lr=arg.lrate,
	weight_decay=arg.l2
)

fp = open(arg.results, 'w')

model = fit_model(
	model,
	train=train_loader,
	test=test_loader,
	optimizer=optimizer,
	criterion=criterion,
	device=device,
	epochs=arg.epochs,
	file=fp
)
fp.close()

# save the model
torch.save(model.state_dict(), arg.modelsave)

	






















