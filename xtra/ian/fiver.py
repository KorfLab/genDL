
import math
import sys

import gendl.seqio

def get_seqs(filename):
	seqs = []
	for seq in gendl.seqio.read_raw(filename):
		seqs.append(seq)
	return seqs

def count_substr(pat, seqs):
	count_left = 0
	count_right = 0
	for seq in seqs:
		if pat in seq[0:20]: count_left += 1
		if pat in seq[22:]: count_right += 1
	return count_left, count_right
		

# file of trues, file of fakes
assert(len(sys.argv) == 3)
trues = get_seqs(sys.argv[1])
fakes = get_seqs(sys.argv[2])

c5, c3 = count_substr('ACGTA', trues)
score = math.log2(c5/c3)
print(f'{score:.4f}')

c5, c3 = count_substr('ACGTA', fakes)
score = math.log2(c5/c3)
print(f'{score:.4f}')