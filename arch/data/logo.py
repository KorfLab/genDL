#!/usr/bin/env python3
"""
Plot sequence logos for splice sites
"""

import sys
import os
import re
import gzip

import numpy as np
import pandas as pd
import seqlogo
import data_shaper as ds

path = os.path.abspath(os.path.dirname(__file__))
figs_path = os.path.join(path, '../figs')
assert(os.path.isdir(figs_path))

def seq_stats(filename):
	sstats = np.zeros((42,4))
	counter = 0
	for seq in ds.read_data(filename, 42):
		counter += 1
		for i in range(len(seq)):
			if seq[i] == 'A':
				sstats[i,0]+=1
			elif seq[i] == 'C':
				sstats[i,1]+=1
			elif seq[i] == 'G':
				sstats[i,2] += 1
			else:
				sstats[i,3] += 1
				
	sstats = sstats/counter
	return sstats 

def entropy(stats):
	height = dict()
	for pos in stats.keys():
		h = 0
		for nt in stats[pos].keys():
			h += stats[pos][nt]*math.log2(stats[pos][nt])
		
		h = -1.0*h
		print(h)
		for nt in stats[pos].keys():
			if pos in height:
				height[pos][nt] = (2.0-h)*stats[pos][nt]
			else:
				height[pos] = dict()
				height[pos][nt] = (2.0-h)*stats[pos][nt]
	
	return height

if __name__ == '__main__':
	
	for filename in os.listdir(os.getcwd()):
		if re.search(r'hilo', filename):
			continue
		
		if re.search(r'gz$', filename):
			print(filename)
			f = filename.split('.')
			name = f[0]+'.'+f[1]+'.'+'logo.png'
			logofile = figs_path+'/'+name
			stats = seq_stats(filename)
#			print(stats)
			ppm = seqlogo.Ppm(stats)
			seqlogo.seqlogo(ppm, ic_scale=True, format='png', size='large',
				filename=logofile,stacks_per_line=45)