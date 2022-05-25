
import argparse
import random
import statistics
import sys

from gendl import pwm, seqio

if __name__ == '__main__':

	parser = argparse.ArgumentParser(
		description='Evaluate the performance of simple PWM methods')
	parser.add_argument('--file1', required=True, type=str,
		metavar='<file>', help='fasta file of observed sites')
	parser.add_argument('--file0', required=True, type=str,
		metavar='<file>', help='fasta file of not observed sites')
	parser.add_argument('--xvalid', required=False, type=int, default=4,
		metavar='<int>', help='x-fold cross-validation [%(default)s]')
	parser.add_argument('--limit', required=False, type=int,
		metavar='<int>', help='limit the data set size by this amount')
	parser.add_argument('--seed', required=False, type=int,
		metavar='<int>', help='random seed')
	parser.add_argument('--noisy', action='store_true', help='see progress')
	arg = parser.parse_args()

	if arg.seed: random.seed(arg.seed)

	# read sequences and reformat
	seqs1 = [(1, seq) for name, seq in seqio.read_fasta(arg.file1)]
	seqs0 = [(0, seq) for name, seq in seqio.read_fasta(arg.file0)]
	if arg.limit: seqs = seqs1[:arg.limit] + seqs0[:arg.limit]
	else:         seqs = seqs1 + seqs0
	random.shuffle(seqs)

	# cross-validation splitting
	accs = []
	hs = []
	for train, test in seqio.cross_validation(seqs, arg.xvalid):

		# make pwms from seqs
		trues = [seq for label, seq in train if label == 1]
		fakes = [seq for label, seq in train if label == 0]
		tpwm = pwm.make_pwm(trues)
		fpwm = pwm.make_pwm(fakes)

		# score vs. test set
		tp, tn, fp, fn = 0, 0, 0, 0
		for entry in test:
			label, seq = entry
			tscore = pwm.score_pwm(tpwm, seq)
			fscore = pwm.score_pwm(fpwm, seq)

			if label == 1:
				if tscore > fscore: tp += 1
				else:               fn += 1
			else:
				if fscore > tscore: tn += 1
				else:               fp += 1
		acc = (tp + tn) / (tp + tn + fp + fn)
		h = pwm.entropy(tpwm)
		accs.append(acc)
		hs.append(h)
		if arg.noisy: print(tp, tn, fp, fn, acc, h, file=sys.stderr)

	print(statistics.mean(accs))
