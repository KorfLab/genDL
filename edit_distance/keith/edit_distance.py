#!/usr/bin/env python3

import argparse
import sys
import gzip
from math import floor

parser = argparse.ArgumentParser(description=''.join(('Compute edit distance',
	'between true and fake for acceptor/donor sequences')))
parser.add_argument('--true', required=True, type=str,
    metavar='<str>', help='true donor/acceptor sequences')
parser.add_argument('--fake', required=True, type=str,
	metavar='<str>', help='fake donor/acceptor sequences')
parser.add_argument('--type', required=True, type=str,
	metavar='<str>', help='acceptor or donor')
	
arg = parser.parse_args()


def read_data(filename, limit):
	fp = None
	count=0
	seqs = []
	if filename.endswith('.gz'):
		fp = gzip.open(filename, 'rt')
	else:
		fp = open(filename)

	for line in fp.readlines():
		line = line.rstrip()
		if len(line) != 42: continue
		seqs.append(line)
		count+=1
		if count == limit:
			break
		
	fp.close()
	return seqs

def edit_distance(seq1, seq2):
	distance = 0
#	print(seq1)
#	print(seq2)
	for i in range(len(seq1)):
		if seq1[i] != seq2[i]:
			distance += 1
	return distance
	
true = read_data(arg.true, 10000)
fake = read_data(arg.fake, 10000)

#ed = dict()
min = 50
minpair = ('', '')
count = 0
size = floor(len(true)*0.05)
print(size)
print('01234567890123456789')
for seqt in true:
#	ed[seqt] = dict()
	for seqf in fake:
		dis = edit_distance(seqt, seqf)
#		print(dis)
#		ed[seqt][seqf] = dis
		if dis < min:
			min = dis
			minpair = (seqt, seqf)
	count+=1

	if count % size == 0:
		print('=',sep=' ',end='',flush=True)

print()
print(min, minpair)