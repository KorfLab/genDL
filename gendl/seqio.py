import gzip
import random
import sys

def read_fasta(filename):
	"""
	Function that reads fasta files

	Keyword arguments:
	filename -- path to the file, type = str 

	Returns sequence name followed by sequences from the imported file
	"""

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
	"""
  	Function that converts sequences stored in fasta format into one-hot encoded data
  
  	Keywords:
  	file -- path to the fasta file containing sequences, type = str
  	label -- label provided by the use (example: 0, 1, 2, 3 ...), type = int
  
  	Returns one-hot encoded sequences
  	"""
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
	"""
  	Function that returns sequencing data in fasta format
  
  	Keywords:
  	filename -- path to the fasta file containing sequences, type = str
  
  	Returns unfiltered names and sequences
  	"""
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
	"""
  	Function that generates random dna sequences based on desired length (not weighted)
  
  	Keywords:
  	length -- length of generated dna sequence provided by the user, type = int
  
  	Returns random dna sequence based on input length, type = str
  	"""
	nts = 'ACGT'
	seq = []
	for i in range(length):
		seq.append(random.choice(nts))
	return ''.join(seq)

def cross_validation(seqs, x):
	"""
  	Function that generated train and test set of a provided dataset
  
  	Keywords:
  	seqs -- list of sequences, type = list
  	x -- number of cross validations, type = int
  
  	Returns train and test sets, type = list, list
  	"""
	for i in range(x):
		train = []
		test = []
		for j in range(len(seqs)):
			if j % x == i: test.append(seqs[j])
			else:          train.append(seqs[j])
		yield train, test

