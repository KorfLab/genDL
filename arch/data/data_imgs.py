#!/usr/bin/env python3

import argparse
import sys
import os
import re
import gzip
import pickle

import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.colors
import data_shaper as ds
import numpy as np

parser = argparse.ArgumentParser(description=''.join(('Plot one-hot encoded',
	' sequence matrices')))
parser.add_argument('-n', required=False, type=int, default=10,
    metavar='<str>', help='number of sequences to grab')

path = os.path.abspath(os.path.dirname(__file__))
figs_path = os.path.join(path, '../figs')
assert(os.path.isdir(figs_path))

def data_setup(num):
	
	acc = dict()
	acc['hi'] = dict()
	acc['lo'] = dict()
	acc['fake'] = dict()
	don = dict()
	don['hi'] = dict()
	don['lo'] = dict()
	don['fake'] = dict()

	for filename in os.listdir(os.getcwd()):
		if re.search(r'hilo', filename):
			continue
		
		if re.search(r'gz$', filename):
			print(filename)
			f = filename.split('.')
			onehot = ds.one_hotter(filename, num, 42)
			if f[0] == 'acc':
				if f[1] == 'fake':
					acc['fake'] = onehot
				elif f[1] == 'hi':
					acc['hi'] = onehot
				else:
					acc['lo'] = onehot
			else:
				if f[1] == 'fake':
					don['fake'] = onehot
				elif f[1] == 'hi':
					don['hi'] = onehot
				else:
					don['lo'] = onehot
	
	return acc, don

def seq_matrix(data,num,savepath):
	fig = plt.figure(figsize=(5*(63/(4*num)),5))
	gs = gridspec.GridSpec(num,3,
		wspace=0.0,hspace=0.0)
	
	hc = 0
	lc = 0
	fc = 0

	cg = matplotlib.colors.ListedColormap(["white","green"])
	cr = matplotlib.colors.ListedColormap(["white","red"])
	cp = matplotlib.colors.ListedColormap(["white","purple"])

	for i in range(num):
		for j in range(3):
			ax = plt.subplot(gs[i,j])
			if j < 1:
				ax.imshow(np.transpose(data['hi'][hc,:,:]), cmap=cg)
				ax.tick_params(axis=u'both',which=u'both',length=0)
				ax.set_xticklabels([])
				ax.set_yticks(np.arange(0,4,1))
				ax.set_yticklabels(['A','C','G','T'],fontsize=8,fontweight=1)
				ax.set_aspect('auto')
				hc+=1
				continue
			elif j < 2:
				ax.imshow(np.transpose(data['lo'][lc,:,:]), cmap=cr)
				ax.tick_params(axis=u'both',which=u'both',length=0)
				ax.set_xticklabels([])
				ax.set_yticklabels([])
				ax.set_aspect('auto')
				lc+=1
				continue
			else:
				ax.imshow(np.transpose(data['fake'][fc,:,:]), cmap=cp)
				ax.tick_params(axis=u'both',which=u'both',length=0)
				ax.set_xticklabels([])
				ax.set_yticklabels([])
				ax.set_aspect('auto')
				fc+=1
				continue
	
	plt.savefig(savepath)
	return

if __name__ == '__main__':
	
	arg = parser.parse_args()
	
	acc, don = data_setup(arg.n)
	
	seq_matrix(acc, arg.n, figs_path+'/acc_onehot_examples.png')
	seq_matrix(don, arg.n, figs_path+'/don_one_hot_examples.png')