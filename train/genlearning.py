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
parser.add_argument('--type', required=True, type=str,
	metavar='<str>', help='acceptor or donor')

arg = parser.parse_args()


true = pickle.load(open(arg.true, "rb"))
fake = pickle.load(open(arg.fake, "rb"))
seqs = np.concatenate((true, fake), axis=0)
print(seqs.shape)

size = true.shape[0]
truelabel = np.array([[0]]*size, dtype=np.uint8)
fakelabel = np.array([[1]]*size, dtype=np.uint8)
labels = np.concatenate((truelabel, fakelabel), axis=0)
print(labels.shape)
labels = keras.utils.to_categorical(labels)

print(truelabel.shape, fakelabel.shape)

model = keras.Sequential([
	keras.layers.Flatten(input_shape=(42,4)),
	keras.layers.Dense(82, activation='relu'),
	keras.layers.Dense(42, activation='relu'),
	keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
	loss=tf.keras.losses.CategoricalCrossentropy(),
	metrics=['accuracy'])
	
model.fit(seqs, labels, epochs=10, batch_size=1, validation_split=0.20)
"""
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

place all features together 
all labels together



#tensorboard = TensorBoard(log_dir='logs/')
"""