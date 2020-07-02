#!/usr/bin/env python3

import argparse
import sys
import os
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
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

arg = parser.parse_args()

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

model = keras.Sequential([
	keras.layers.Flatten(input_shape=(42,4)),
#	keras.layers.Dense(200, activation='elu',
#		kernel_regularizer=tf.keras.regularizers.l2(0.001)),
#	keras.layers.Dropout(0.50),
#	keras.layers.Dense(42, activation='elu',
#		kernel_regularizer=tf.keras.regularizers.l2(0.001)),
#	keras.layers.Dropout(0.50),
#	keras.layers.Dense(21, activation='elu',
#		kernel_regularizer=tf.keras.regularizers.l2(0.001)),
#	keras.layers.Dropout(0.50),
#	keras.layers.Dense(10, activation='elu',
#		kernel_regularizer=tf.keras.regularizers.l2(0.001)),
#	keras.layers.Dropout(0.25),
#	keras.layers.Dense(5, activation='elu',
#		kernel_regularizer=tf.keras.regularizers.l2(0.001)),
#	keras.layers.Dropout(0.25),
	keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),
	loss=tf.keras.losses.BinaryCrossentropy(),
	metrics=['binary_accuracy', tf.keras.metrics.TruePositives(),
	tf.keras.metrics.FalseNegatives(), tf.keras.metrics.TrueNegatives(),
	tf.keras.metrics.FalsePositives()])
	
model.fit(seqs_train, labels_train, epochs=10, batch_size=100, 
	validation_data=(seqs_val, labels_val))

loss, acc, tp, fn, tn, fp = model.evaluate(seqs_val, labels_val, batch_size=1)

tpr = tp/(tp+fn)
tnr = tn/(tn+fp)
ppv = tp/(tp+fp)
npv = tn/(tn+fn)
print(f"{tpr:.4f} {tnr:.4f} {ppv:.4f} {npv:.4f}")
print(f"{(tpr+ppv)/2:.4f}")

#tensorboard = TensorBoard(log_dir='logs/')