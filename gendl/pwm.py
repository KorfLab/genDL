
def make_pwm(seqs):
	length = len(seqs[0])
	
	# create counts
	count = []
	for i in range(length): count.append({'A':0, 'C':0, 'G':0, 'T':0})
	total = 0
	for i in range(len(seqs)):
		seq = seqs[i]
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

def score_pwm(pwm, seq):
	p = 1
	for i in range(len(seq)):
		p *= pwm[i][seq[i]]
	return p

def display_pwm(pwm):
	for i in range(len(pwm)):
		print(f'{str(i)}\t{pwm[i]["A"]:.3f}\t{pwm[i]["C"]:.3f}\t{pwm[i]["G"]:.3f}\t{pwm[i]["T"]:.3f}')
