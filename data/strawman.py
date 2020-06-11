
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

def get_seqs(file, limit):
	seqs = []
	with gzip.open(file, 'rt') as fp:
		for line in fp.readlines():
			seqs.append(line.rstrip())
			if len(seqs) == limit: break
	return seqs

def make_pwm(seqs, boost=None):

	length = len(seqs[0])

	# create counts
	count = []
	for i in range(length): count.append({'A':0, 'C':0, 'G':0, 'T':0})
	total = 0
	for i in range(len(seqs)):
		seq = seqs[i]
		total += 1
		for i in range(len(seq)):
			if boost == None:
				count[i][seq[i]] += 1
			else:
				count[i][seq[i]] += boost[i]

	# create freqs
	freq = []
	for i in range(length): freq.append({})
	for i in range(length):
		for c in count[i]:
			freq[i][c] = count[i][c] / total

	return freq

def make_kmer(seqs, k, boost=None):
	length = len(seqs[0])
	count = {}
	total = 0
	for seq in seqs:
		for i in range(length - k):
			kmer = seq[i:i+k]
			if kmer not in count: count[kmer] = 1
			else:                 count[kmer] += 1
			total += 1
	
	if len(count) != 4 ** k: return None # some zero counts found
	
	freq = {}
	for kmer in count: freq[kmer] = count[kmer] / total
	return freq

def score_pwm(seq, pwm):
	p = 1
	for i in range(len(seq)):
		p *= pwm[i][seq[i]]
	return p

def score_kmer(seq, mm, k):
	p = 1
	for i in range(len(seq) - k):
		kmer = seq[i:i+k]
		p *= mm[kmer]
	return p

def kmer_threshold(trues, fakes, xv):

	sys.stderr.write('\nkmer_threshold\n')
	length = len(trues[0])
	k = 0
	kmers_full = True
	max_k_acc = 0
	max_k = None
	while kmers_full:
		k += 1
		sum_acc = 0
		
		for x in range(xv):
		
			# collect testing and training sets
			train, test = [], []
			for i in range(len(trues)):
				if i % xv == x: test.append(trues[i])
				else:           train.append(trues[i])
	
			# build mm
			km = make_kmer(train, k)
			if km == None:
				kmers_full = False
				break
				
			# find maximum thresholds
			vmax = 0
			for kmer in km:
				if km[kmer] > vmax: vmax = km[kmer]
			tmax = vmax ** length
			tmin = (0.25 ** (k)) ** length
			
			# find max t by stepping down tmax by half each time
			sys.stderr.write(f'set-{x} k-{k}')
			t = tmax
			n = 0
			acc_max = 0
			while True:
				t /= 2
				if t < tmin: break # no sense going below tmin
			
				# score against test set
				tp, tn, fp, fn = 0, 0, 0, 0
				for seq in test:
					s = score_kmer(seq, km, k)
					if s > t: tp += 1
					else:     fn += 1
				for i in range(len(test)): # fakes could be bigger, so limit
					seq = fakes[i]
					s = score_kmer(seq, km, k)
					if s > t: fp += 1
					else:     tn += 1
	
				if tp:
					tpr = tp / (tp + fn)
					ppv = tp / (tp + fp)
					acc = (tpr + ppv) / 2
					if acc > acc_max: acc_max = acc
			sys.stderr.write(f' {acc_max}\n')
			sum_acc += acc_max
			if sum_acc > max_k_acc:
				max_k_acc = sum_acc
				max_k = k
	
	return max_k_acc/xv

def pwm_threshold(trues, fakes, xv):

	sys.stderr.write('\npwm_threshold\n')
	sum_acc = 0
	for x in range(xv):

		# collect testing and training sets
		train, test = [], []
		for i in range(len(trues)):
			if i % xv == x: test.append(trues[i])
			else:           train.append(trues[i])
	
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
				s = score_pwm(seq, pwm)
				if s > t: tp += 1
				else:     fn += 1
			for i in range(len(test)): # fakes could be bigger, so limit
				seq = fakes[i]
				s = score_pwm(seq, pwm)
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
		sys.stderr.write(f' {t} {acc_max}\n')
		sum_acc += acc_max
		
	return sum_acc / xv

def pwm_vs_pwm(trues, fakes, xv):

	sys.stderr.write('\npwm_vs_pwm\n')
	TPR, TNR, PPV, NPV = 0, 0, 0, 0
	for x in range(xv):

		# collect testing and training sets
		ttrain, ttest, ftrain, ftest = [], [], [], []
		for i in range(len(trues)):
			if i % xv == x: ttest.append(trues[i])
			else:           ttrain.append(trues[i])
		for i in range(len(fakes)):
			if i % xv == x: ftest.append(fakes[i])
			else:           ftrain.append(fakes[i])

		# make pwms
		tpwm = make_pwm(ttrain)
		fpwm = make_pwm(ftrain)
		
		# score against test set
		tp, tn, fp, fn = 0, 0, 0, 0
		for seq in ttest:
			ts = score_pwm(seq, tpwm)
			fs = score_pwm(seq, fpwm)
			if ts > fs: tp += 1
			else:       fn += 1

		for seq in ftest:
			ts = score_pwm(seq, tpwm)
			fs = score_pwm(seq, fpwm)
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

	return (TPR+PPV)/(xv*2)

def boosted_pwms(trues, fakes, xv):

	sys.stderr.write('\nboosted_pwms\n')
	TPR, TNR, PPV, NPV = 0, 0, 0, 0
	for x in range(xv):

		# collect testing and training sets
		ttrain, ttest, ftrain, ftest = [], [], [], []
		for i in range(len(trues)):
			if i % xv == x: ttest.append(trues[i])
			else:           ttrain.append(trues[i])
		for i in range(len(fakes)):
			if i % xv == x: ftest.append(fakes[i])
			else:           ftrain.append(fakes[i])

		# make pwms without boosting
		tpwm = make_pwm(ttrain)
		fpwm = make_pwm(ftrain)
		
		# score against test set and set boost
		tboost = []
		fboost = []
		tp, tn, fp, fn = 0, 0, 0, 0
		for seq in ttest:
			ts = score_pwm(seq, tpwm)
			fs = score_pwm(seq, fpwm)
			if ts > fs:
				tp += 1
				tboost.append(1)
			else:
				fn += 1
				tboost.append(2)
		for seq in ftest:
			ts = score_pwm(seq, tpwm)
			fs = score_pwm(seq, fpwm)
			if ts > fs:
				fp += 1
				fboost.append(2)
			else:
				tn += 1
				fboost.append(1)
				
		# re-make pwms with boosting
		tpwm = make_pwm(ttrain, boost=tboost)
		fpwm = make_pwm(ftrain, boost=fboost)
		
		# evaluate boosted PWMs
		tp, tn, fp, fn = 0, 0, 0, 0
		for seq in ttest:
			ts = score_pwm(seq, tpwm)
			fs = score_pwm(seq, fpwm)
			if ts > fs: tp += 1
			else:       fn += 1
		for seq in ftest:
			ts = score_pwm(seq, tpwm)
			fs = score_pwm(seq, fpwm)
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

	return (TPR+PPV)/(xv*2)

if __name__ == '__main__':

	trues = get_seqs(arg.true, arg.n)
	fakes = get_seqs(arg.fake, arg.n)
	
	acc1 = pwm_threshold(trues, fakes, arg.x)	
	print(f'PWM Threshold: {acc1:.4f}')
	
	acc2 = pwm_vs_pwm(trues, fakes, arg.x)
	print(f'PWM vs. PWM: {acc2:.4f}')
	
#	acc3 = boosted_pwms(trues, fakes, arg.x)
#	print(f'Boosted PWMs: {acc3:.4f}')
	
#	acc4 = kmer_threshold(trues, fakes, arg.x)
#	print(f'KMER Threshold: {acc4:.4f}')

