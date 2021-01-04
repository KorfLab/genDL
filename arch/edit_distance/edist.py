#!/usr/bin/env python3

import sys

def edit_dist(s1, s2):
	d = 0
	for i in range(len(s1)):
		if s1[i] != s2[i]: d += 1
	return d

seqs = []
with open(sys.argv[1]) as fp:
	for line in fp.readlines():
		line = line.rstrip()
		seqs.append(line)

min = 50
comps = 0
for i in range(len(seqs)):
	for j in range(i+1, len(seqs)):
		comps += 1
		d = edit_dist(seqs[i], seqs[j])
		if d < min: min = d
print(min, comps)
		

"""
	1	2	3	4
1	.	x	x	x
2		.	x	x
3			.	x
4				.

Cat meow

Unix 10     \n
Mac 13      \r
Win 13 10

"""
