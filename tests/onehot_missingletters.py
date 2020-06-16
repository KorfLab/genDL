#!/usr/bin/env python3

import argparse
import sys
import os
import gzip
import pickle
import random

import numpy as np
from tensorflow import keras

parser = argparse.ArgumentParser(description=''.join(('Test to_categorical',
	'function with sequences that contain only subset of ACGT')))
parser.add_argument('--sequences', required=True, type=str,
    metavar='<str>', help='test sequences')

arg = parser.parse_args()

dna_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
int_dna = {0: 'A', 1: 'C', 2: 'G', 3: 'G'}

window = 42

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
	return np.array(intlist, dtype=np.uint8)

def seqs_list(filename):
	seqs = list()
	for seq in read_data(filename):
		seqs.append(to_integer(seq))
	
	return seqs


def one_hotter(file):
	seqs = seqs_list(file)
	random.shuffle(seqs)

	array = np.zeros((len(seqs), window, 4), dtype=np.float64)
	for i, seq in enumerate(seqs):
		counts={0:0,1:0,2:0,3:0}
		letters = ''
		for pos in seq:
			letters+=int_dna[pos]
		print(letters)
		print(seq)
		encoded = keras.utils.to_categorical(seq, 
											 num_classes=4,
											 dtype=np.float64)
		print(encoded)
		array[i,:,:] = encoded
	
	return array

if __name__ == '__main__':
	enc = one_hotter(arg.sequences)
	print('')
	print(enc)