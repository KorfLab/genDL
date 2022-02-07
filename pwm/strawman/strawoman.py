
import argparse
import random
import statistics
import sys

from gendl import pwm, seqio


def rotate(s, x):
	return s

def max_score(model, seq):
	w = len(model)
	max = 0
	for i in range(len(seq) -w +1):
		s = pwm.score_pwm(model, seq[i:i+w])
		if s > max: max = s
	return max
	#print(seq[i:i+w], s)
	#print(model, seq)
	#sys.exit(0)
	#pwm.score_pwm(tpwm, seq)

if __name__ == '__main__':

	parser = argparse.ArgumentParser(
		description='Evaluate the performance of simple PWM methods')
	parser.add_argument('--file1', required=True, type=str,
		metavar='<file>', help='fasta file of observed sites')
	parser.add_argument('--file0', required=True, type=str,
		metavar='<file>', help='fasta file of not observed sites')
	parser.add_argument('--focus', required=True, type=int, nargs=2,
		metavar='<int>', help='motif position')
	#parser.add_argument('--blur', required=True, type=int,
	#	metavar='<int>', help='change position of motif')
	parser.add_argument('--xvalid', required=False, type=int, default=4,
		metavar='<int>', help='x-fold cross-validation [%(default)s]')
	parser.add_argument('--seed', required=False, type=int,
		metavar='<int>', help='random seed')
	arg = parser.parse_args()

	if arg.seed: random.seed(arg.seed)
	
	fa, fb = arg.focus

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
		trues = [seq[fa:fb] for label, seq in train if label == 1]
		fakes = [seq[fa:fb] for label, seq in train if label == 0]
		tpwm = pwm.make_pwm(trues)
		fpwm = pwm.make_pwm(fakes)

		# score vs. test set
		tp, tn, fp, fn = 0, 0, 0, 0
		for entry in test:
			label, seq = entry
			tscore = max_score(tpwm, seq)
			fscore = max_score(fpwm, seq)

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
		print(tp, tn, fp, fn, acc, h)

	print(statistics.mean(accs), statistics.mean(hs))
