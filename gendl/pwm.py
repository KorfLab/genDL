
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

def score_pwm(seq, pwm):
	p = 1
	for i in range(len(seq)):
		p *= pwm[i][seq[i]]
	return p

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


