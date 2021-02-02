import gzip
import random
import sys

def linereader(filename):
	"""
	*Generator that returns files line by line with minimal memory*
	
	Removes line endings also.

	**Parameters:**
	_______________

	+ filename -- path to the file (str)
	"""
	
	fp = None
	if   filename == '-':          fp = sys.stdin
	elif filename.endswith('.gz'): fp = gzip.open(filename, 'rt')
	else:                          fp = open(filename)
	
	while True:
		line = fp.readline()
		if line == '': break
		else: yield line.rstrip()
	
	fp.close()
	
def read_fasta(filename):
	"""
	*Generator that returns records from a fasta file*

	*Returns a tuple of (name, seq)*

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

	for line in linereader(filename):
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
	*Converts sequences stored in fasta format into one-hot encoded data*

	*Returns one-hot encoded sequences with the label on the end*

	**Parameter:**
	______________

	+ file -- path to the fasta file containing sequences (str)
	+ label -- label provided by the user (int)
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

def fasta2binary(file, label):
	"""
	*Converts sequences stored in fasta format into binary encoded data*

	*Returns binary encoded sequences with the label on the end*

	**Parameter:**
	______________

	+ file -- path to the fasta file containing sequences (str)
	+ label -- label provided by the user (int)
	"""

	data = []
	for name, seq in read_fasta(file):
		s = ''
		for nt in seq:
			if   nt == 'A': s += '00'
			elif nt == 'C': s += '01'
			elif nt == 'G': s += '10'
			elif nt == 'T': s += '11'
			else: raise()
		s += str(label)
		data.append(s)
	return data

def random_dna(length):
	"""
	*Function that generates random dna sequences based on desired length (not weighted)*

	*Returns random dna sequence based on input length (str)*

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
	*Function that generated train and test set of a provided dataset*

	*Returns train and test sets (list), (list)*

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

