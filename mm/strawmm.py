
import argparse
import random
import statistics
import sys

from gendl import pwm, seqio

def make_mm(seqs, k):
	count = {}
	total = 0
	for seq in seqs:
		for i in range(len(seq) -k + 1):
			kmer = seq[i:i+k]
			if kmer not in count: count[kmer] = 0
			count[kmer] += 1
			total += 1
	freq = {}
	for kmer in count:
		freq[kmer] = count[kmer] / total
	return freq

def score_mm(mm, seq, k):
	score = 1
	for i in range(len(seq) -k +1):
		kmer = seq[i:i+k]
		score *= mm[kmer]
	return score

if __name__ == '__main__':

	parser = argparse.ArgumentParser(
		description='Evaluate the performance of simple Markov model')
	parser.add_argument('--file1', required=True, type=str,
		metavar='<file>', help='fasta file of observed sites')
	parser.add_argument('--file0', required=True, type=str,
		metavar='<file>', help='fasta file of not observed sites')
	parser.add_argument('--k', required=False, type=int, default=4,
		metavar='<int>', help='kmer size [%(default)i]')
	parser.add_argument('--xvalid', required=False, type=int, default=4,
		metavar='<int>', help='x-fold cross-validation [%(default)s]')
	parser.add_argument('--seed', required=False, type=int,
		metavar='<int>', help='random seed')
	arg = parser.parse_args()

	if arg.seed: random.seed(arg.seed)

	# read sequences and reformat
	seqs1 = [(1, seq) for name, seq in seqio.read_fasta(arg.file1)]
	seqs0 = [(0, seq) for name, seq in seqio.read_fasta(arg.file0)]
	seqs = seqs1 + seqs0
	random.shuffle(seqs) # just in case for real data

	# cross-validation splitting
	accs = []
	hs = []
	for train, test in seqio.cross_validation(seqs, arg.xvalid):

		# make pwms from seqs
		trues = [seq for label, seq in train if label == 1]
		fakes = [seq for label, seq in train if label == 0]
		tmm = make_mm(trues, arg.k)
		fmm = make_mm(fakes, arg.k)

		# score vs. test set
		tp, tn, fp, fn = 0, 0, 0, 0
		for entry in test:
			label, seq = entry
			tscore = score_mm(tmm, seq, arg.k)
			fscore = score_mm(fmm, seq, arg.k)

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

