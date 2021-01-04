#!/usr/bin/env python3

import sys
import edik
import time

def edit_distance(s1, s2):
	n = 0
	for i in range(len(s1)):
		if s1[i] != s2[i]: n += 1
	return n

seqs = []
with open(sys.argv[1]) as fp:
	for line in fp.readlines():
		seqs.append(line)

t0 = time.perf_counter()
sum = 0
count = 0
for i in range(len(seqs)):
	for j in range(i+1, len(seqs)):
		sum += edit_distance(seqs[i], seqs[j])
		count += 1
print(sum/count)
t1 = time.perf_counter()
ptime = t1 - t0
print('python', ptime)

t0 = time.perf_counter()
sum = 0
count = 0
for i in range(len(seqs)):
	for j in range(i+1, len(seqs)):
		sum += edik.edit_distance(seqs[i], seqs[j])
		count += 1
print(sum/count)
t1 = time.perf_counter()
ctime = t1 - t0
print('cython', ctime)
print('speed up', ptime / ctime)
