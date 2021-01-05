
import argparse
import random
import sys

from gendl import pwm, seqio

parser = argparse.ArgumentParser(
	description='Evaluate the performance of simple PWM methods')
parser.add_argument('--file1', required=True, type=str,
	metavar='<file>', help='fasta file')
parser.add_argument('--file2', required=True, type=str,
	metavar='<file>', help='fasta file')
parser.add_argument('--xvalid', required=False, type=int, default=4,
	metavar='<int>', help='x-fold cross-validation [%(default)s]')
parser.add_argument('--seed', required=False, type=int,
	metavar='<int>', help='random seed')
arg = parser.parse_args()

if arg.seed: random.seed(arg.seed)

"""
def pwm_vs_threshold(pwm, t, tsites, fsites):
	tp, tn, fp, fn = 0, 0, 0, 0

	for seq in tsites:
		s = score_pwm(seq, pwm)
		if s > t: tp += 1
		else:     fn += 1

	for seq in fsites:
		s = score_pwm(seq, pwm)
		if s > t: fp += 1
		else:     tn += 1

	return tp, tn, fp, fn

def pwm_vs_pwm(tpwm, fpwm, trues, fakes):
	tp, tn, fp, fn = 0, 0, 0, 0

	for seq in trues:
		ts = score_pwm(seq, tpwm)
		fs = score_pwm(seq, fpwm)
		if ts > fs: tp += 1
		else:       fn += 1

	for seq in fakes:
		ts = score_pwm(seq, tpwm)
		fs = score_pwm(seq, fpwm)
		if ts > fs: fp += 1
		else:       tn += 1

	return tp, tn, fp, fn
"""


# read sequences
seqs1 = [(1, seq) for name, seq in seqio.read_fasta(arg.file1)]
seqs2 = [(0, seq) for name, seq in seqio.read_fasta(arg.file2)]
seqs = seqs1 + seqs2
random.shuffle(seqs)

# cross-validation splitting
for train, test in seqio.cross_validation(seqs, arg.xvalid):
	tseqs = []
	fseqs = []
	for entry in train:
		label, seq = entry
		if label == 1: tseqs.append(seq)
		else:          fseqs.append(seq)
	
	tpwm = pwm.make_pwm(tseqs)
	fpwm = pwm.make_pwm(fseqs)
	
	pwm.display_pwm(tpwm)
	sys.exit()
	
	# stopped here
	
	