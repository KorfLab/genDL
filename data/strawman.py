
import argparse
import sys
import gzip

parser = argparse.ArgumentParser(
	description='PWM-based discriminator.')

parser.add_argument('--true', required=True, type=str,
	metavar='<path>', help='.gz file of true sequences')
parser.add_argument('--fake', required=True, type=str,
	metavar='<path>', help='.gz file of fake sequences')
parser.add_argument('--n', required=False, type=int, default=1000,
	metavar='<int>', help='maximum number of sequences to use [%(default)i]')
parser.add_argument('--x', required=False, type=int, default=2,
	metavar='<int>', help='cross-validation level [%(default)i]')

arg = parser.parse_args()

def make_pwm(seqs, boost=None):

	length = len(seqs[0])

	# create counts
	count = []
	for i in range(length): count.append({'A':0, 'C':0, 'G':0, 'T':0})
	total = 0
	for seq in seqs:
		total += 1
		for i in range(len(seq)):
			count[i][seq[i]] += 1			

	# create freqs
	freq = []
	for i in range(length): freq.append({})
	for i in range(length):
		for c in count[i]:
			freq[i][c] = count[i][c] / total

	return freq

def get_seqs(file, limit):
	seqs = []
	with gzip.open(file, 'rt') as fp:
		for line in fp.readlines():
			seqs.append(line.rstrip())
			if len(seqs) == limit: break
	return seqs

def score_seq(seq, pwm):
	p = 1
	for i in range(len(seq)):
		p *= pwm[i][seq[i]]
	return p

trues = get_seqs(arg.true, arg.n)
fakes = get_seqs(arg.fake, arg.n)

# Experiment 1: finding an optimal threshold
print('Experiment 1: optimal threshold')
ACC = 0
for x in range(arg.x):

	# collect testing and training sets
	train, test = [], []
	for i in range(len(trues)):
		if i % arg.x == x: test.append(trues[i])
		else:              train.append(trues[i])

	# build pwm
	pwm = make_pwm(train)

	# find maximum threshold
	tmax = 1
	for i in range(len(pwm)):
		max = 0
		for c in pwm[i]:
			if pwm[i][c] > max: max = pwm[i][c]
		tmax *= max
	tmin = 0.25 ** len(pwm)
	
	# find max t by stepping down tmax by half each time
	sys.stderr.write(f'set-{x} ')
	t = tmax
	n = 0
	acc_max = 0
	while True:
		t /= 2
		if t < tmin: break # no sense going below tmin
		
		# score against test set
		tp, tn, fp, fn = 0, 0, 0, 0
		for seq in test:
			s = score_seq(seq, pwm)
			if s > t: tp += 1
			else:     fn += 1
		for i in range(len(test)): # limit the number of fakes to size of test
			seq = fakes[i]
			s = score_seq(seq, pwm)
			if s > t: fp += 1
			else:     tn += 1

		if tp:
			tpr = tp / (tp + fn)
			ppv = tp / (tp + fp)
			acc = (tpr + ppv) / 2
			if acc > acc_max: acc_max = acc
			sys.stderr.write(f'x')
		else:
			sys.stderr.write(f'.')
	sys.stderr.write(f' {t} {acc}\n')
	ACC += acc
	
print(f'Accuracy: {ACC/arg.x:.4f}')

# Experiment 2: competing PWMs
print('Experiment 2: competing PWMs')
TPR, TNR, PPV, NPV = 0, 0, 0, 0
for x in range(arg.x):

	# collect testing and training sets
	ttrain, ttest, ftrain, ftest = [], [], [], []
	for i in range(len(trues)):
		if i % arg.x == x: ttest.append(trues[i])
		else:              ttrain.append(trues[i])
	for i in range(len(fakes)):
		if i % arg.x == x: ftest.append(fakes[i])
		else:              ftrain.append(fakes[i])

	# make pwms
	tpwm = make_pwm(ttrain)
	fpwm = make_pwm(ftrain)
		
	# score against test set
	tp, tn, fp, fn = 0, 0, 0, 0
	for seq in ttest:
		ts = score_seq(seq, tpwm)
		fs = score_seq(seq, fpwm)
		if ts > fs: tp += 1
		else:       fn += 1

	for seq in ftest:
		ts = score_seq(seq, tpwm)
		fs = score_seq(seq, fpwm)
		if ts > fs: fp += 1
		else:       tn += 1

	# gather performance stats
	tpr = tp / (tp + fn)
	tnr = tn / (tn + fp)
	ppv = tp / (tp + fp)
	npv = tn / (tn + fn)

	sys.stderr.write(f'set-{x} {tpr:.3f} {tnr:.3f} {ppv:.3f} {npv:.3f}\n')
	TPR += tpr
	TNR += tnr
	PPV += ppv
	NPV += npv

print(f'Accuracy: {(TPR+PPV)/(arg.x * 2):.4f}')


