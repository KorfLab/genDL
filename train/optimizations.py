#!/usr/bin/env python3

import argparse
import sys
import os
import pickle
import math
import time
import json

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import tensorflow as tf
from ff_models import FeedForwardModel
import prep
import eval

parser = argparse.ArgumentParser(description=''.join(('Feed-forward NN ',
	'optimization for learning acceptor/donor splice site labels')))
parser.add_argument('--true', required=True, type=str,
    metavar='<str>', help='true donor/acceptor pickled one-hot sequences')
parser.add_argument('--fake', required=True, type=str,
	metavar='<str>', help='fake donor/acceptor pickled one-hot sequences')
parser.add_argument('--params', required=False, type=str,
	metavar='<str>', help='json file for model parameters')
parser.add_argument('--val', required=False, type=float,
	metavar='<str>', help='acceptor or donor')
parser.add_argument('--xv', required=False, type=int,
	metavar='<int>', help='number of cross-validation folds')

arg = parser.parse_args()

def listprod(list=[],level=0,sublist=[],funnel=True):
	"""
	Inputs:
	
	Outputs:
	"""
	
	if not list:
		yield []
		return
			
	if level+1 == len(list):
		for k in list[level]:
			if funnel:
				if len(sublist) != 0:
					if k >= sublist[level-1]: continue
			sublist.append(k)
			yield sublist
			sublist.pop()
		return
	elif level+1 < len(list):
		for k in list[level]:
			if funnel:
				if len(sublist) != 0:
					if k >= sublist[level-1]: continue
			sublist.append(k)
			for l in listprod(list=list,level=level+1,sublist=sublist,funnel=funnel):
				yield(l)
			sublist.pop()


""" Define model parameters """

with open(arg.params) as json_file:
	params = json.load(json_file)

""" pull in data """

if arg.val:
	X, y, xv, yv = prep.val_split(arg.true, arg.fake, arg.val)

""" Make Models """
counter = 0
t0 = time.perf_counter()
for l in layers:
	print('layers ',l)
	
	for s in listprod(list=layers_list[:l],level=0,sublist=[],funnel=True):
		print('\t',s)
		counter += 1
		model = FeedForwardModel(
			layers=l,sizes=s,reg=[],dropout=[]).build()

		model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),
			loss=tf.keras.losses.BinaryCrossentropy(),
			metrics=['binary_accuracy', tf.keras.metrics.TruePositives(),
			tf.keras.metrics.FalseNegatives(), tf.keras.metrics.TrueNegatives(),
			tf.keras.metrics.FalsePositives()])
		
		model.fit(X, y, epochs=10, batch_size=500, 
			validation_data=(xv, yv),verbose=2)
			
		loss,acc,tp,fn,tn,fp = model.evaluate(xv, yv, batch_size=1)

		tpr = tp/(tp+fn)
		tnr = tn/(tn+fp)
		ppv = tp/(tp+fp)
		npv = tn/(tn+fn)
		print(f"{tpr:.4f} {tnr:.4f} {ppv:.4f} {npv:.4f}")
		print(f"{(tpr+ppv)/2:.4f}")
		
		results[counter] = dict()
		results[counter]['model'] = f'layers: {l} sizes: {s}'
		results[counter]['TPR'] = tpr
		results[counter]['TNR'] = tnr
		results[counter]['PPV'] = ppv
		results[counter]['NPV'] = npv
		results[counter]['FSC'] = (2.0*tpr*ppv)/(tpr+ppv)
		
#		saving[counter] = (model, model.fsc)

t1 = time.perf_counter()
totaltime = round(t1-t0,4)

#update send saving to results.txt

print()
print(eval.summary(results, totaltime))