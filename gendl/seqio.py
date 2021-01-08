import gzip
import random
import sys

def read_fasta(filename):
	name = None
	seqs = []
	
	fp = None
	if filename == '-':
		fp = sys.stdin
	elif filename.endswith('.gz'):
		fp = gzip.open(filename, 'rt')
	else:
		fp = open(filename)

	for line in fp.readlines():
		line = line.rstrip()
		if line.startswith('>'):
			if len(seqs) > 0:
				seq = ''.join(seqs)
				yield(name, seq)
				name = line[1:]
				seqs = []
			else:
				name = line[1:]
		else:
			seqs.append(line)
	yield(name, ''.join(seqs))
	fp.close()

def fasta2onehot(file, label):
	data = []
	for name, seq in read_fasta(file):
		s = ''
		for nt in seq:
			if   nt == 'A': s += '1000'
			elif nt == 'C': s += '0100'
			elif nt == 'G': s += '0010'
			elif nt == 'T': s += '0001'
			else: raise()
		s += str(label)
		data.append(s)
	return data

def read_raw(filename):
	fp = None
	if filename == '-':
		fp = sys.stdin
	elif filename.endswith('.gz'):
		fp = gzip.open(filename, 'rt')
	else:
		fp = open(filename)
		
	for line in fp.readlines():
		line = line.rstrip()
		yield(line)
	
	fp.close()

def random_dna(length):
	nts = 'ACGT'
	seq = []
	for i in range(length):
		seq.append(random.choice(nts))
	return ''.join(seq)

def cross_validation(seqs, x):
	for i in range(x):
		train = []
		test = []
		for j in range(len(seqs)):
			if j % x == i: test.append(seqs[j])
			else:          train.append(seqs[j])
		yield train, test

