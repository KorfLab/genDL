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

if arg.val:
	X, y, xv, yv = prep.val_split(arg.true, arg.fake, arg.val)
	print(X.shape)
	X = X.reshape(X.shape[0],51,4,1)
	xv = xv.reshape(xv.shape[0],51,4,1)
	print(X.shape)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(600, (5,2), activation='elu', 
	input_shape=(51, 4, 1), 
	kernel_regularizer=tf.keras.regularizers.l2(1e-2)))
model.add(tf.keras.layers.Dropout(0.50))
model.add(tf.keras.layers.MaxPooling2D((2, 1)))

model.add(tf.keras.layers.Conv2D(300, (4,1), activation='elu',
	kernel_regularizer=tf.keras.regularizers.l2(1e-2)))
model.add(tf.keras.layers.Dropout(0.50))
model.add(tf.keras.layers.MaxPooling2D((2, 1)))

model.add(tf.keras.layers.Conv2D(120, (4,2), activation='elu',
	kernel_regularizer=tf.keras.regularizers.l2(1e-2)))
model.add(tf.keras.layers.Dropout(0.50))
model.add(tf.keras.layers.MaxPooling2D((2, 1)))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
	loss=tf.keras.losses.BinaryCrossentropy(),
	metrics=['binary_accuracy', tf.keras.metrics.TruePositives(),
	tf.keras.metrics.FalseNegatives(), tf.keras.metrics.TrueNegatives(),
	tf.keras.metrics.FalsePositives()])

model.summary()


model.fit(X, y, epochs=20, batch_size=1000, 
	validation_data=(xv, yv),verbose=2)
	
loss,acc,tp,fn,tn,fp = model.evaluate(xv, yv, batch_size=1)