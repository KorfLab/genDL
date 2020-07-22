#!/usr/bin/env python3

import argparse
import sys
import os
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser(description=''.join(('Learning acceptor/donor',
	'splice site labels')))
parser.add_argument('--true', required=True, type=str,
    metavar='<str>', help='true donor/acceptor pickled one-hot sequences')
parser.add_argument('--fake', required=True, type=str,
	metavar='<str>', help='fake donor/acceptor pickled one-hot sequences')
parser.add_argument('--val', required=False, type=float, default=0.10,
	metavar='<str>', help='acceptor or donor')

arg = parser.parse_args()

class FeedForwardModel():

	def __init__(self,
				 layers=0,
				 sizes=[0],
				 **kwargs):
		
		self.layers = layers
		self.sizes = sizes
		
		if kwargs:
			if kwargs['dropout']:
				assert(len(kwargs['dropout']) != 0)
				self.dropout = kwargs['dropout']
			else:
				self.dropout = []
		
			if kwargs['reg']:
				assert(len(kwargs['reg']) != 0)
				self.reg = kwargs['reg']
			else:
				self.reg = []
		
		self.model = self.build()
		

	def build(self):
		model = tf.keras.Sequential()
		model.add(tf.keras.layers.Flatten(input_shape=(42,4)))
		
		for i in range(self.layers):
			model.add(tf.keras.layers.Dense(
				self.sizes[i],
				activation='elu',
				kernel_regularizer=tf.keras.regularizers.l2(self.reg[i]))
			)
			model.add(tf.keras.layers.Dropout(self.dropout[i]))
		
		model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
		
		return model

if __name__ == '__main__':
	def sizing(val, size):
		train = int(size*(1-val))
		val   = int(size*val)
		assert(train+val == size)
		return train, val
	
	true = pickle.load(open(arg.true, "rb"))
	fake = pickle.load(open(arg.fake, "rb"))

	train_size, val_size = sizing(arg.val, true.shape[0])

	np.random.shuffle(true)
	np.random.shuffle(fake)

	true_train = true[:train_size,:,:]
	true_val   = true[train_size:,:,:]

	fake_train = fake[:train_size,:,:]
	fake_val   = fake[train_size:,:,:]

	seqs_train = np.concatenate((true_train, fake_train), axis=0)
	seqs_val   = np.concatenate((true_val, fake_val), axis=0)
	print(seqs_train.shape, seqs_val.shape)

	truelabel_train = np.array([[1]]*train_size, dtype=np.uint8)
	truelabel_val   = np.array([[1]]*val_size, dtype=np.uint8)
	fakelabel_train = np.array([[0]]*train_size, dtype=np.uint8)
	fakelabel_val   = np.array([[0]]*val_size, dtype=np.uint8)


	labels_train = np.concatenate((truelabel_train, fakelabel_train), axis=0)
	labels_val   = np.concatenate((truelabel_val, fakelabel_val), axis=0)
	print(labels_train.shape, labels_val.shape)
	
	model1 = FeedForwardModel(
		layers=0,sizes=[],reg=[],dropout=[]).build()
	
	model2 = FeedForwardModel(
		layers=1,sizes=[84],reg=[0],dropout=[0]).build()

	model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),
		loss=tf.keras.losses.BinaryCrossentropy(),
		metrics=['binary_accuracy', tf.keras.metrics.TruePositives(),
		tf.keras.metrics.FalseNegatives(), tf.keras.metrics.TrueNegatives(),
		tf.keras.metrics.FalsePositives()])
	
	model1.fit(seqs_train, labels_train, epochs=10, batch_size=100, 
		validation_data=(seqs_val, labels_val),verbose=2)

	loss,acc,tp,fn,tn,fp = model1.evaluate(seqs_val, labels_val, batch_size=1)

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
	
	model2.fit(seqs_train, labels_train, epochs=10, batch_size=100, 
		validation_data=(seqs_val, labels_val),verbose=2)

	loss,acc,tp,fn,tn,fp = model2.evaluate(seqs_val, labels_val, batch_size=1)

	tpr = tp/(tp+fn)
	tnr = tn/(tn+fp)
	ppv = tp/(tp+fp)
	npv = tn/(tn+fn)
	print(f"{tpr:.4f} {tnr:.4f} {ppv:.4f} {npv:.4f}")
	print(f"{(tpr+ppv)/2:.4f}")
	
	
	layer_dictionary = {0: [], 1: [300, 100, 50], 2:[10, 15, 20]}
	
	for layers in layer_dictionary.keys():
		for nodes in layers:
			model = FeedForward().build()
			model.compile()
			model.fit()
	
	hparams
			
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	