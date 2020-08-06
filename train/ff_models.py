#!/usr/bin/env python3

import argparse
import sys
import os
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import tensorflow as tf
import prep

class FeedForwardModel():

	def __init__(self,
				 layers=0,
				 sizes=[0],
				 **kwargs):
		
		self.layers = layers
		self.sizes = sizes
		
		if kwargs:
			if kwargs['dropout']:
				assert(len(kwargs['dropout']) == self.layers)
				self.dropout = kwargs['dropout']
			else:
				self.dropout = []
		
			if kwargs['reg']:
				assert(len(kwargs['reg']) == self.layers)
				self.reg = kwargs['reg']
			else:
				self.reg = []
		
		self.model = self.build()
		

	def build(self):
		model = tf.keras.Sequential()
		model.add(tf.keras.layers.Flatten(input_shape=(42,4)))
		
		for i in range(self.layers):
			if self.reg:
				model.add(tf.keras.layers.Dense(
					self.sizes[i],
					activation='elu',
					kernel_regularizer=tf.keras.regularizers.l2(self.reg[i]))
				)
			else:
				model.add(tf.keras.layers.Dense(
					self.sizes[i],
					activation='elu')
				)
			
			if self.dropout:
				model.add(tf.keras.layers.Dropout(self.dropout[i]))	
					
		model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
		
		return model

if __name__ == '__main__':
	
	import prep
	
	parser = argparse.ArgumentParser(description=''.join(('Learning acceptor/',
		'donor splice site labels')))
	parser.add_argument('--true', required=True, type=str,
    	metavar='<str>', help='true donor/acceptor pickled one-hot sequences')
	parser.add_argument('--fake', required=True, type=str,
		metavar='<str>', help='fake donor/acceptor pickled one-hot sequences')
	parser.add_argument('--val', required=False, type=float, default=0.10,
		metavar='<str>', help='acceptor or donor')

	arg = parser.parse_args()
	
	X, y, vx, vy = prep.val_split(arg.true, arg.fake, arg.val)
	
	model1 = FeedForwardModel(
		layers=0,sizes=[],reg=[],dropout=[]).build()
	
	model2 = FeedForwardModel(
		layers=1,sizes=[84],reg=[0],dropout=[0]).build()

	model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),
		loss=tf.keras.losses.BinaryCrossentropy(),
		metrics=['binary_accuracy', tf.keras.metrics.TruePositives(),
		tf.keras.metrics.FalseNegatives(), tf.keras.metrics.TrueNegatives(),
		tf.keras.metrics.FalsePositives()])
	
	model1.fit(X, y, epochs=10, batch_size=100, 
		validation_data=(vx, vy),verbose=2)

	loss,acc,tp,fn,tn,fp = model1.evaluate(vx, vy, batch_size=1)
	
	tpr = tp/(tp+fn)
	tnr = tn/(tn+fp)
	ppv = tp/(tp+fp)
	npv = tn/(tn+fn)
	print(f"{tpr:.4f} {tnr:.4f} {ppv:.4f} {npv:.4f}")
	print(f"{(tpr+ppv)/2:.4f}")
	
	model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),
		loss=tf.keras.losses.BinaryCrossentropy(),
		metrics=['binary_accuracy', tf.keras.metrics.TruePositives(),
		tf.keras.metrics.FalseNegatives(), tf.keras.metrics.TrueNegatives(),
		tf.keras.metrics.FalsePositives()])
	
	model2.fit(X, y, epochs=10, batch_size=100, 
		validation_data=(vx, vy),verbose=2)

	loss,acc,tp,fn,tn,fp = model2.evaluate(vx, vy, batch_size=1)

	tpr = tp/(tp+fn)
	tnr = tn/(tn+fp)
	ppv = tp/(tp+fp)
	npv = tn/(tn+fn)
	print(f"{tpr:.4f} {tnr:.4f} {ppv:.4f} {npv:.4f}")
	print(f"{(tpr+ppv)/2:.4f}")