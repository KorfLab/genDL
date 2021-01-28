
import math
import sys

def make_pwm(seqs):
	"""
	Function for making position weight matrix
	
	**Parameters:**
	
	+ seqs -- list of sequences (list)
	
	**Returns:**
	
	+ a position weight matrix (list of dictionaries)
	"""
	
	length = len(seqs[0])

	# create count data structure
	count = [ {'A':0, 'C':0, 'G':0, 'T':0} for i in range(length) ]
	
	# do the counting
	total = 0
	for seq in seqs:
		total += 1
		for i in range(len(seq)):
			count[i][seq[i]] += 1

	# create freqs
	freq = [ {} for i in range(length) ]
	for i in range(length):
		for c in count[i]:
			freq[i][c] = count[i][c] / total

	return freq

def score_pwm(pwm, seq):
	"""
	*Function to show scoring against the created position weight matrix* <br/>
	
	*Returns a score for the sequence scored against pwm* <br/>
	
	**Parameters:**
	_______________
	+ pwm -- position weight matrix (list of dictionaries) <br/>
	  	** for help refer to make_pwm function
	+ seq -- a single sequence (int)
  	"""
	p = 1
	for i in range(len(seq)):
		p *= pwm[i][seq[i]]
	return p

def display_pwm(pwm):
	"""
  	*Function that displays the probability of each nucleotide in a given position of the pwm* <br/>
	
	*Returns a pwm in a readable format* <br/>

	**Parameters:**
	_______________
  	+ pwm -- position weight matrix (list of dictionaries) <br/>
  		** for help refer to make_pwm function
  	"""

	for i in range(len(pwm)):
		print(f'{str(i)}\t{pwm[i]["A"]:.3f}\t{pwm[i]["C"]:.3f}\t{pwm[i]["G"]:.3f}\t{pwm[i]["T"]:.3f}')

def entropy(pwm):
	"""
	*Function that shows how the randomness of the chosen base in a given position* <br/>
  	
  	*Returns an entropy score (float)* <br/>

	**Parameters:**
	_______________

	+ pwm -- posiiton weight matrix (list of dictionaries) <br/>
  		** for help refer to make_pwm function
  	"""

	H = 0

	for i in range(len(pwm)):
		h = 0
		for nt in pwm[i]:
			if pwm[i][nt] != 0: h += pwm[i][nt] * math.log2(pwm[i][nt])
		H += 2 + h

	return H
