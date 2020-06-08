#!/usr/bin/env python3

import argparse
import sys
import os
import gzip
import pickle

import numpy as np
from tensorflow import keras

parser = argparse.ArgumentParser(description=''.join(('Shape splice-site data,',
	'into one-hot encoding matrices')))
parser.add_argument('--true', required=True, type=str,
    metavar='<str>', help='true donor/acceptor data')
parser.add_argument('--fake', required=True, type=str,
	metavar='<str>', help='fake donor/acceptor data')
parser.add_argument('--type', required=True, type=str,
	metavar='<str>', help='acceptor or donor examples')
parser.add_argument('--size', required=False, type=int, default=10000,
	metavar='<str>', help='number of examples to process')

arg = parser.parse_args()

assert(os.path.isdir('/'.join((os.getcwd(), 'one_hot'))))
assert(arg.type == 'acceptor' or arg.type == 'donor')

dna_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

def read_data(filename):
	fp = None

	if filename.endswith('.gz'):
		fp = gzip.open(filename, 'rt')
	else:
		fp = open(filename)

	for line in fp.readlines():
		line = line.rstrip()
		yield(line)
		
	fp.close()

def to_integer(seq):
	intlist = list()
	for nt in seq:
		intlist.append(dna_int[nt])
	return np.array(intlist, dtype=np.float64)

def pickling(type, bool, size, data):
	filepath = f"{os.getcwd()}/one_hot/{type}.{bool}.{size}.pickle"
	print(filepath)
	pickle_out = open(filepath, "wb")
	pickle.dump(data, pickle_out)
	pickle_out.close()
	
def one_hotter(file, size):
	counter = 0
	array = np.zeros((size, 42, 4), dtype=np.float64)
	for seq in read_data(file):
		seqdata = to_integer(seq)
		encoded = keras.utils.to_categorical(seqdata)
		array[counter,:,:] = encoded
		counter+=1
		if counter >= arg.size:
			break
	return array
	
true = one_hotter(arg.true, arg.size)
fake = one_hotter(arg.fake, arg.size)

pickling(arg.type, 'true', arg.size, true)
pickling(arg.type, 'fake', arg.size, fake)