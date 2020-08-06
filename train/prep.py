#!/usr/bin/env python3

import sys
import os
import pickle

import numpy as np

def sizing(val, size):
	train = int(size*(1-val))
	val   = int(size*val)
	assert(train+val == size)
	return train, val

def val_split(true_pickle=None, fake_pickle=None, val=None):
	assert(true_pickle is not None)
	assert(fake_pickle is not None)
	assert(val is not None)	
	
	true = pickle.load(open(true_pickle, "rb"))
	fake = pickle.load(open(fake_pickle, "rb"))
	
	assert(true.shape[0] == fake.shape[0])
	
	train_size, val_size = sizing(val, true.shape[0])

	np.random.shuffle(true)
	np.random.shuffle(fake)

	true_train = true[:train_size,:,:]
	true_val   = true[train_size:,:,:]

	fake_train = fake[:train_size,:,:]
	fake_val   = fake[train_size:,:,:]

	seqs_train = np.concatenate((true_train, fake_train), axis=0)
	seqs_val   = np.concatenate((true_val, fake_val), axis=0)

	truelabel_train = np.array([[1]]*train_size, dtype=np.uint8)
	truelabel_val   = np.array([[1]]*val_size, dtype=np.uint8)
	fakelabel_train = np.array([[0]]*train_size, dtype=np.uint8)
	fakelabel_val   = np.array([[0]]*val_size, dtype=np.uint8)


	labels_train = np.concatenate((truelabel_train, fakelabel_train), axis=0)
	labels_val   = np.concatenate((truelabel_val, fakelabel_val), axis=0)
	
	return seqs_train, labels_train, seqs_val, labels_val
	
def cross_validation(true, fake, xv):
	assert(xv > 1)
	assert(type(xv) is int)
	
	step = int((1/xv)*len(true))
	train_size = true.shape[0]-step
	val_size = step
	
	np.random.shuffle(true)
	np.random.shuffle(fake)
	
	for i in range(0,len(true)-step+1,step):
#		print(i, i+step, step, len(true))
		
		true_val = true[i:i+step,:,:]
		true_train = np.concatenate((true[0:i,:,:],true[i+step:,:,:]), axis=0)
		
		fake_val = fake[i:i+step,:,:]
		fake_train = np.concatenate((fake[0:i,:,:],fake[i+step:,:,:]), axis=0)
		
		seqs_train = np.concatenate((true_train, fake_train), axis=0)
		seqs_val   = np.concatenate((true_val, fake_val), axis=0)
		
		truelabel_train = np.array([[1]]*train_size, dtype=np.uint8)
		truelabel_val   = np.array([[1]]*val_size, dtype=np.uint8)
		fakelabel_train = np.array([[0]]*train_size, dtype=np.uint8)
		fakelabel_val   = np.array([[0]]*val_size, dtype=np.uint8)
		
		labels_train = np.concatenate((truelabel_train,fakelabel_train),axis=0)
		labels_val   = np.concatenate((truelabel_val, fakelabel_val), axis=0)
		
		yield(seqs_train, labels_train, seqs_val, labels_val)
		
if __name__ == '__main__':
	
	import argparse
	
	os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
	
	import tensorflow as tf
	from tensorflow import keras
	from tensorflow.keras.callbacks import TensorBoard
	
	
	parser = argparse.ArgumentParser(description=''.join(('Learning acceptor/donor',
		'splice site labels')))
	parser.add_argument('--true', required=True, type=str,
    	metavar='<str>', help='true donor/acceptor pickled one-hot sequences')
	parser.add_argument('--fake', required=True, type=str,
		metavar='<str>', help='fake donor/acceptor pickled one-hot sequences')
	parser.add_argument('--val', required=False, type=float, default=0.10,
		metavar='<str>', help='acceptor or donor')
	parser.add_argument('--xv', required=False, type=int,
		metavar='<int>', help='number of cross-validation folds')

	arg = parser.parse_args()
	
	true = pickle.load(open(arg.true, "rb"))
	fake = pickle.load(open(arg.fake, "rb"))
	
	assert(true.shape[0] == fake.shape[0])
	
	if arg.xv:
 		tpr = 0
 		tnr = 0
 		ppv = 0
 		npv = 0
 		
 		for X, y, vx, vy in cross_validation(true, fake, arg.xv):
 			print(X.shape, y.shape, vx.shape, vy.shape)
 			model = keras.Sequential([
 				keras.layers.Flatten(input_shape=(42,4)),
 				keras.layers.Dense(84, activation='elu'),
 				keras.layers.Dense(1, activation='sigmoid')
 			])
 			
 			model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),
 				loss=tf.keras.losses.BinaryCrossentropy(),
 				metrics=['binary_accuracy', tf.keras.metrics.TruePositives(),
 				tf.keras.metrics.FalseNegatives(), tf.keras.metrics.TrueNegatives(),
 				tf.keras.metrics.FalsePositives()])
 				
 			model.fit(X, y, epochs=50, batch_size=10, 
 				validation_data=(vx, vy))
 			
 			loss, acc, tp, fn, tn, fp = model.evaluate(vx, vy, batch_size=1)
 			
 			tpr += tp/(tp+fn)
 			tnr += tn/(tn+fp)
 			ppv += tp/(tp+fp)
 			npv += tn/(tn+fn)
 		
 		print(f"{tpr/arg.xv:.4f} {tnr/arg.xv:.4f} {ppv/arg.xv:.4f} {npv/arg.xv:.4f}")
 		print(f"{(tpr+ppv)/(2*arg.xv):.4f}")
 		
	else:
		X, y, vx, vy = val_split(arg.true, arg.fake, arg.val)
		
		model = keras.Sequential([
 			keras.layers.Flatten(input_shape=(42,4)),
 			keras.layers.Dense(84, activation='elu'),
 			keras.layers.Dense(1, activation='sigmoid')
 		])
 		
 		model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),
 			loss=tf.keras.losses.BinaryCrossentropy(),
 			metrics=['binary_accuracy', tf.keras.metrics.TruePositives(),
 			tf.keras.metrics.FalseNegatives(), tf.keras.metrics.TrueNegatives(),
 			tf.keras.metrics.FalsePositives()])
 		
 		model.fit(X, y, epochs=50, batch_size=10, 
 			validation_data=(vx, vy))
 		
 		loss, acc, tp, fn, tn, fp = model.evaluate(vx, vy, batch_size=1)
 		
 		tpr += tp/(tp+fn)
 		tnr += tn/(tn+fp)
 		ppv += tp/(tp+fp)
 		npv += tn/(tn+fn)
 		
 		print(f"{tpr/arg.xv:.4f} {tnr/arg.xv:.4f} {ppv/arg.xv:.4f} {npv/arg.xv:.4f}")
 		print(f"{(tpr+ppv)/(2*arg.xv):.4f}")