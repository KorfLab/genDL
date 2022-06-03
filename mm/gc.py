import argparse
import random
import statistics
import sys

from gendl import pwm, seqio

def make_gc(seqs):
	count = {'AT': 0, 'CG': 0}
	total = 0
	for seq in seqs:
		for nt in seq:
			if nt == 'A' or nt == 'T': count['AT'] += 1
			else:                      count['CG'] += 1
			total += 1
	freq = {}
	for nt in count:
		freq[nt] = count[nt] / total
	return freq

def score_gc(gc, seq):
	score = 1
	for nt in seq:
		if nt == 'A' or nt == 'T': score *= gc['AT']
		else:                      score *= gc['CG']
	return score

# CLI
parser = argparse.ArgumentParser(
	description='Evaluate GC for exons/introns')
parser.add_argument('--exons', required=True, type=str,
	metavar='<file>', help='fasta file of observed sites')
parser.add_argument('--introns', required=True, type=str,
	metavar='<file>', help='fasta file of not observed sites')
parser.add_argument('--xvalid', required=False, type=int, default=4,
	metavar='<int>', help='x-fold cross-validation [%(default)s]')
parser.add_argument('--seed', required=False, type=int,
	metavar='<int>', help='random seed')
arg = parser.parse_args()

if arg.seed: random.seed(arg.seed)

# read sequences and reformat
seqs1 = [(1, seq) for name, seq in seqio.read_fasta(arg.exons)]
seqs0 = [(0, seq) for name, seq in seqio.read_fasta(arg.introns)]
seqs = seqs1 + seqs0
random.shuffle(seqs)

# cross-validation splitting
accs = []
hs = []
for train, test in seqio.cross_validation(seqs, arg.xvalid):

	# make pwms from seqs
	trues = [seq for label, seq in train if label == 1]
	fakes = [seq for label, seq in train if label == 0]
	tm = make_gc(trues)
	fm = make_gc(fakes)
	
	print(tm)

	# score vs. test set
	tp, tn, fp, fn = 0, 0, 0, 0
	for entry in test:
		label, seq = entry
		tscore = score_gc(tm, seq)
		fscore = score_gc(fm, seq)

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

