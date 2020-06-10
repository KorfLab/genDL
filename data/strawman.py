
import sys
import json
import gzip
import math
import random

count = []
for i in range(42): count.append({'A':0, 'C':0, 'G':0, 'T':0})
total = 0
with gzip.open(sys.argv[1], 'rt') as fp:
	for line in fp.readlines():
		total += 1
		for i in range(len(line) -1):
			c = line[i]
			count[i][c] += 1			

freq = []
for i in range(42): freq.append({})
for i in range(42):
	for c in count[i]:
		freq[i][c] = count[i][c] / total

exp = 0.25 ** 42
t = 10
hi, lo, = 0,0
with gzip.open(sys.argv[2], 'rt') as fp:
	for line in fp.readlines():
		line = line.rstrip()
		obs = 1
		for i in range(len(line)):
			c = line[i]
			obs *= freq[i][c]
		score = math.log2(obs/exp)
		if score > t: hi += 1
		else: lo += 1

print(hi / (hi + lo))

###
hi, lo, = 0,0
for i in range(10000):
	seq = []
	for j in range(20): seq.append(random.choice('ACGT'))
	seq.append('A')
	seq.append('G')
	for j in range(20): seq.append(random.choice('ACGT'))
	obs = 1
	for i in range(len(seq)):
		c = seq[i]
		obs *= freq[i][c]
	score = math.log2(obs/exp)
	if score > t: hi += 1
	else: lo += 1
	#print(''.join(seq))
	
print(hi / (hi + lo))



