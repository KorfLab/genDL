#!/usr/bin/env python3

import argparse
import sys
import os
import json

import grimoire.toolbox as toolbox
import grimoire.hmm as hmm
from grimoire.sequence import DNA
from grimoire.feature import Feature, FeatureTable
from grimoire.genome import Reader

def write_file(name, seqs):
	with open(name, 'w') as fp:
		for seq in seqs:
			fp.write(seq)
			fp.write('\n')


threshold = 1000
flank = 20
genome = Reader(fasta='wb276.fa', gff='wb276.gff3')
sd_hi = {}   # splice donors above threshold
sd_lo = {}   # splice donors below threshold
sd_fake = {} # fake splice donors
sa_hi = {}   # acceptors
sa_lo = {}   # acceptors
sa_fake = {} # acceptors

for chrom in genome:
	seq = chrom.seq.upper()
	
	# true splice donors and acceptors
	seen = {}
	for f in chrom.ftable.features:
		if f.strand == '-': continue # training set is + strand
		if f.source == 'RNASeq_splice':
			db, de = f.beg-(flank+1), f.beg+flank+1
			ab, ae = f.end-(flank+2), f.end+flank
			don = seq[db:de]
			acc = seq[ab:ae]
			seen[don] = True
			seen[acc] = True
			if f.score >= threshold:
				sd_hi[don] = True
				sa_hi[acc] = True
			else:
				sd_lo[don] = True
				sa_lo[acc] = True

	# fake splice donors and acceptors
	# 	must be transcribed (inside a gene)
	#	must match GT..AG consensus
	#	must not be annotated
	for f in chrom.ftable.features:		
		if f.type == 'gene' and f.strand == '+':
			for i in range(f.beg, f.end):
				n2 = seq[i:i+2]
				if n2 == 'GT':
					db, de = i-flank, i+flank+2
					don = seq[db:de]
					if don not in seen:
						sd_fake[don] = True
				elif n2 == 'AG':
					ab, ae = i-20, i+22
					acc = seq[ab:ae]
					if acc not in seen:
						sa_fake[acc] = True

# outputs
write_file('don.lo.true.txt', sd_lo.keys())
write_file('don.hi.true.txt', sd_hi.keys())
write_file('acc.lo.true.txt', sa_lo.keys())
write_file('acc.hi.true.txt', sa_hi.keys())
write_file('don.fake.txt', sd_fake.keys())
write_file('acc.fake.txt', sa_fake.keys())
