#!/usr/bin/env python3

import argparse
import datetime
import sys
import os
import gzip
import pickle
import random

import numpy as np
from tensorflow import keras

parser = argparse.ArgumentParser(description=''.join(('Shape splice-site data',
	' into one-hot encoding matrices')))
parser.add_argument('--true', required=True, type=str,
    metavar='<str>', help='true donor/acceptor data')
parser.add_argument('--fake', required=True, type=str,
	metavar='<str>', help='fake donor/acceptor data')
parser.add_argument('--type', required=True, type=str,
	metavar='<str>', help='acceptor or donor examples')
parser.add_argument('--level', required=True, type=str,
	metavar='<str>', help='hi, lo, or hilo')
parser.add_argument('--number', required=False, type=int, default=1000,
	metavar='<int>', help='number of examples desired. -1 if you want all data')
parser.add_argument('--window', required=False, type=int, default=42,
	metavar='<int>', help='size of sequences')

dna_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
int_dna = {0: 'A', 1: 'C', 2: 'G', 3: 'G'}

def read_data(filename, win):
	fp = None

	if filename.endswith('.gz'):
		fp = gzip.open(filename, 'rt')
	else:
		fp = open(filename)

	for line in fp.readlines():
		line = line.rstrip()
		assert(len(line) == win)
		yield(line)
		
	fp.close()

def to_integer(seq):
	intlist = list()
	for nt in seq:
		intlist.append(dna_int[nt])
	return np.array(intlist, dtype=np.uint8)

def seqs_list(filename, win):
	seqs = list()
	for seq in read_data(filename, win):
		seqs.append(to_integer(seq))
	
	return seqs

def pickling(type, bool, size, level, data):
	if size == -1:
		size = 'all'
	time = datetime.datetime.now().strftime('%H:%M:%S')
	filepath = f"{os.getcwd()}/one_hot/{type}.{bool}.{level}.{size}.{time}.pickle"
	print(filepath)
	pickle_out = open(filepath, "wb")
	pickle.dump(data, pickle_out)
	pickle_out.close()
	
def one_hotter(file, size, win):
	seqs = seqs_list(file, win)
	random.shuffle(seqs)
	if size == -1:
		arg.number = len(seqs)

	array = np.zeros((size, win, 4), dtype=np.float64)
	for i, seq in enumerate(seqs[:size]):
		encoded = keras.utils.to_categorical(seq, 
											 num_classes=4,
											 dtype=np.float64)
		
		array[i,:,:] = encoded
	
	return array

if __name__ == '__main__':

	arg = parser.parse_args()
	
	assert(os.path.isdir('/'.join((os.getcwd(), 'one_hot'))))
	assert(arg.type == 'acceptor' or arg.type == 'donor')
	assert(arg.level == 'hi' or arg.level == 'lo' or arg.level == 'hilo')	

	true = one_hotter(arg.true, arg.number, arg.window)
	fake = one_hotter(arg.fake, arg.number, arg.window)

	pickling(arg.type, 'true', arg.number, arg.level, true)
	pickling(arg.type, 'fake', arg.number, arg.level, fake)