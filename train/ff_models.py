#!/usr/bin/env python3

import argparse
import sys
import os
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers, regularizers, losses, metrics
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import f1_score, recall_score, precision_score

class Metrics(Callback):
	def __init__(self, validation):   
		super(Metrics, self).__init__()
		self.validation = validation    
		
		print('validation shape', len(self.validation[0]))
		
	def on_train_begin(self, logs={}):        
		self.val_f1s = []
		self.val_recalls = []
		self.val_precisions = []
	
	def on_epoch_end(self, epoch, logs={}):
		val_targ = self.validation[1]   
		val_predict = (np.asarray(self.model.predict(self.validation[0]))).round()
		
		val_f1 = f1_score(val_targ, val_predict)
		val_recall = recall_score(val_targ, val_predict)         
		val_precision = precision_score(val_targ, val_predict)
		
		self.val_f1s.append(round(val_f1, 6))
		self.val_recalls.append(round(val_recall, 6))
		self.val_precisions.append(round(val_precision, 6))
		
		print(f' — val_ppv: {val_precision:.4f}, - val_tpr: {val_recall:.4f} — val_f1: {val_f1:.4f}')


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
			
			if 'lr' in kwargs:
				assert(type(kwargs['lr']) != str)
				self.lr = kwargs['lr']
			else:
				self.lr = 1e-6
		else:
			self.lr = 1e-6
		
		self.model = self.build()
		

	def build(self):
		model = tf.keras.Sequential()
		model.add(layers.Flatten(input_shape=(42,4)))
		
		for i in range(self.layers):
			if self.reg:
				model.add(layers.Dense(
					self.sizes[i],
					activation='elu',
					kernel_regularizer=regularizers.l2(self.reg[i]))
				)
			else:
				model.add(layers.Dense(
					self.sizes[i],
					activation='elu')
				)
			
			if self.dropout:
				model.add(layers.Dropout(self.dropout[i]))	
					
		model.add(layers.Dense(1, activation='sigmoid'))
		
		model.compile(optimizer=optimizers.Adam(learning_rate=self.lr),
			loss=losses.BinaryCrossentropy(), metrics=['binary_accuracy', 
			metrics.TruePositives(name='tp'), metrics.FalseNegatives(name='fn'), 
			metrics.TrueNegatives(name='tn'), metrics.FalsePositives(name='fp'),
			metrics.Recall(name='recall'), 
			metrics.Precision(name='precision')])
		
		return model		

if __name__ == '__main__':
	
	import prep
	import eval
	
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
		layers=0,sizes=[],reg=[],dropout=[])
		#.build()
	
	model2 = FeedForwardModel(
		layers=1,sizes=[84],reg=[0],dropout=[0])
		#.build()
	
	model1.model.fit(X, y, epochs=10, batch_size=100)
#callbacks=[Metrics(validation=(vx,vy))])
	
	r, p, f = eval.performance_metrics(model1.model, vx, vy)
	
	model2.model.fit(X, y, epochs=10, batch_size=1000)
#callbacks=[Metrics(validation=(vx,vy))])
	
	r, p, f = eval.performance_metrics(model2.model, vx, vy)