
import argparse
import itertools
import random
import statistics
import sys

from gendl import pwm, seqio

def make_wam(seqs, order):
	"""
	Function for making weight array matrix
	
	**Parameters:**
	
	+ seqs -- list of sequences (list)
	+ order -- context of Markov model (int)
	
	**Returns:**
	
	+ weight array matrix model (position-context-letter)
	"""
	
	length = len(seqs[0])

	# create the data structures
	alph = 'ACGT'
	count = []
	freq = []
	for i in range(length):
		d = {}
		f = {}
		for tup in itertools.product(alph, repeat=order):
			s = ''.join(tup)
			d[s] = {}
			f[s] = {}
			for nt in alph:
				d[s][nt] = 0
				f[s][nt] = None
		count.append(d)
		freq.append(f)
	
	# do the actual counting
	total = 0
	for seq in seqs:
		total += 1
		for i in range(order, len(seq)):
			ctx = seq[i-order:i]
			nt = seq[i]
			count[i][ctx][nt] += 1
	
	# convert to freqs
	for i in range(length):
		for ctx in count[i]:
			tot = sum(count[i][ctx].values())
			for nt in count[i][ctx]:
				if tot == 0: freq[i][ctx][nt] = 0 # maybe None
				else: freq[i][ctx][nt] = count[i][ctx][nt] / tot
	
	return freq


def score_wam(wam, order, seq):
	"""
	Function for scoring weight array matrix
	
	**Parameters:**
	
	+ wam -- weight array matrix (position-context-letter)
	+ order -- context of Markov model (int)
	+ seq -- sequence (str)
	
	**Returns:**
	
	+ weight array matrix model
	"""
	
	p = 1
	for i in range(order, len(seq)):
		ctx = seq[i-order:i]
		nt = seq[i]
		p *= wam[i][ctx][nt]
	return p
	

if __name__ == '__main__':

	parser = argparse.ArgumentParser(
		description='Evaluate the performance of a WAM method')
	parser.add_argument('--file1', required=True, type=str,
		metavar='<file>', help='fasta file of observed sites')
	parser.add_argument('--file0', required=True, type=str,
		metavar='<file>', help='fasta file of not observed sites')
	parser.add_argument('--order', required=False, type=int, default=1,
		metavar='<int>', help='order of Markov model [%(default)i]')
	parser.add_argument('--xvalid', required=False, type=int, default=4,
		metavar='<int>', help='x-fold cross-validation [%(default)s]')
	parser.add_argument('--seed', required=False, type=int,
		metavar='<int>', help='random seed')
	arg = parser.parse_args()

	if arg.seed: random.seed(arg.seed)
	assert(arg.order >= 1)

	# read sequences and reformat
	seqs1 = [(1, seq) for name, seq in seqio.read_fasta(arg.file1)]
	seqs0 = [(0, seq) for name, seq in seqio.read_fasta(arg.file0)]
	seqs = seqs1 + seqs0
	random.shuffle(seqs) # just in case for real data

	# cross-validation splitting
	accs = []
	for train, test in seqio.cross_validation(seqs, arg.xvalid):

		# make pwms from seqs
		trues = [seq for label, seq in train if label == 1]
		fakes = [seq for label, seq in train if label == 0]
		twam = make_wam(trues, arg.order)
		fwam = make_wam(fakes, arg.order)

		# score vs. test set
		tp, tn, fp, fn = 0, 0, 0, 0
		for entry in test:
			label, seq = entry
			tscore = score_wam(twam, arg.order, seq)
			fscore = score_wam(fwam, arg.order, seq)

			if label == 1:
				if tscore > fscore: tp += 1
				else:               fn += 1
			else:
				if fscore > tscore: tn += 1
				else:               fp += 1
		acc = (tp + tn) / (tp + tn + fp + fn)
		accs.append(acc)
		print(tp, tn, fp, fn, acc)

	print(statistics.mean(accs))
