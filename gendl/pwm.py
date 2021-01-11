
import math
import sys

def make_pwm(seqs):
	"""
	Function for making position weight matrix

	Keyword arguments:
	seqs -- list of sequences, type = list

	Returns a list of dictionaries with frequencies for each base in a given position, type = list of dicionaries
	"""
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
	"""
	Function to show scoring against the created position weight matrix

	Keyword arguments:
	pwm -- position weight matrix, type = list of dictionaries
	  	** for help refer to make_pwm function
	seq -- a single sequence, type = int

	Returns a score for the sequence scored against pwm
  	"""
	p = 1
	for i in range(len(seq)):
		p *= pwm[i][seq[i]]
	return p

def display_pwm(pwm):
	"""
  	Function that displays the probability of each nucleotide in a given position of the pwm

  	Keyword arguments:
  	pwm -- position weight matrix, type = list of dictionaries
  		** for help refer to make_pwm function

  	Returns a pwm in a readable format
  	"""

	for i in range(len(pwm)):
		print(f'{str(i)}\t{pwm[i]["A"]:.3f}\t{pwm[i]["C"]:.3f}\t{pwm[i]["G"]:.3f}\t{pwm[i]["T"]:.3f}')

def entropy(pwm):
	"""
	Function that shows how the randomness of the chosen base in a given position

	Keyword arguments:
	pwm -- posiiton weight matrix, type = list of dictionaries
  		** for help refer to make_pwm function

  	Returns an entropy score, type = float
  	"""

	H = 0

	for i in range(len(pwm)):
		h = 0
		for nt in pwm[i]:
			if pwm[i][nt] != 0: h += pwm[i][nt] * math.log2(pwm[i][nt])
		H += 2 + h

	return H
