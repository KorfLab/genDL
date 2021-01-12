import gzip
import random
import sys

def read_fasta(filename):
	"""
	*Function that reads fasta files* <br/>

	*Returns sequence name followed by sequences from the imported file* <br/>

	**Parameters:**
	_______________

	+ filename -- path to the file (str)
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
	*Function converts sequences stored in fasta format into one-hot encoded data* <br/>

	*Returns one-hot encoded sequences* <br/>

	**Parameter:**
	______________

	+ file -- path to the fasta file containing sequences (str)
	+ label -- label provided by the use (int)
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
	*Function that returns sequencing data in fasta format* <br/>

	*Returns unfiltered names and sequences* <br/>

	**Parameter:**
	______________

	+ filename -- path to the fasta file containing sequences (str)
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
	*Function that generates random dna sequences based on desired length (not weighted)* <br/>

	*Returns random dna sequence based on input length (str)* <br/>

	**Parameter:**
	______________

	+ length -- length of generated dna sequence provided by the user (int)

	"""
	nts = 'ACGT'
	seq = []
	for i in range(length):
		seq.append(random.choice(nts))
	return ''.join(seq)

def cross_validation(seqs, x):
	"""
	*Function that generated train and test set of a provided dataset* <br/>

	*Returns train and test sets (list), (list)* <br/>

	**Parameter:**
	______________
	+ seqs -- list of sequences (list)
	+ x -- number of cross validations (int)
	"""
	for i in range(x):
		train = []
		test = []
		for j in range(len(seqs)):
			if j % x == i: test.append(seqs[j])
			else:          train.append(seqs[j])
		yield train, test

