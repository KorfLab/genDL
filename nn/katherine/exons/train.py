# NN for exon identification
# Usage example:
	# python3 train.py --file1 trues.fa.gz --file0 falses.fa.gz --e 20

import seqio
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
import random  # to scramble rows
import os  # for removing temp file later
import argparse

def create_model(X, layer1=5, layer2=5, layer3=5, layer4=5, layer5=5, layer6=5, layer7=5, layer8=5, layer9=5, layer10=5):  # Using 5 as a placeholder
	NN = keras.Sequential()
	# Hidden layer 1
	NN.add(keras.layers.Dense(layer1, input_dim = X.shape[1], activation='sigmoid'))
	# Dropout layer
	NN.add(keras.layers.Dropout(0.2))
	# L2 regularization
	NN.add(keras.layers.Dense(3, kernel_regularizer='l2'))
	"""
	# Hidden layer 2
	NN.add(keras.layers.Dense(layer2, activation='sigmoid'))
	# Dropout layer
	NN.add(keras.layers.Dropout(0.2))
	# L2 regularization
	NN.add(keras.layers.Dense(3, kernel_regularizer='l2'))
	# Hidden layer 3
	NN.add(keras.layers.Dense(layer3, activation='sigmoid'))
	# Dropout layer
	NN.add(keras.layers.Dropout(0.2))
	# L2 regularization
	NN.add(keras.layers.Dense(3, kernel_regularizer='l2'))
	# Hidden layer 4
	NN.add(keras.layers.Dense(layer4, activation='sigmoid'))
	# Dropout layer
	NN.add(keras.layers.Dropout(0.2))
	# L2 regularization
	NN.add(keras.layers.Dense(3, kernel_regularizer='l2'))
	# Hidden layer 5
	NN.add(keras.layers.Dense(layer4, activation='sigmoid'))
	# Dropout layer
	NN.add(keras.layers.Dropout(0.2))
	# L2 regularization
	NN.add(keras.layers.Dense(3, kernel_regularizer='l2'))
	# Hidden layer 6
	NN.add(keras.layers.Dense(layer4, activation='sigmoid'))
	# Dropout layer
	NN.add(keras.layers.Dropout(0.2))
	# L2 regularization
	NN.add(keras.layers.Dense(3, kernel_regularizer='l2'))
	# Hidden layer 7
	NN.add(keras.layers.Dense(layer4, activation='sigmoid'))
	# Dropout layer
	NN.add(keras.layers.Dropout(0.2))
	# L2 regularization
	NN.add(keras.layers.Dense(3, kernel_regularizer='l2'))
	# Hidden layer 8
	NN.add(keras.layers.Dense(layer4, activation='sigmoid'))
	# Dropout layer
	NN.add(keras.layers.Dropout(0.2))
	# L2 regularization
	NN.add(keras.layers.Dense(3, kernel_regularizer='l2'))
	# Hidden layer 9
	NN.add(keras.layers.Dense(layer4, activation='sigmoid'))
	# Dropout layer
	NN.add(keras.layers.Dropout(0.2))
	# L2 regularization
	NN.add(keras.layers.Dense(3, kernel_regularizer='l2'))
	# Hidden layer 10
	NN.add(keras.layers.Dense(layer4, activation='sigmoid'))
	# Dropout layer
	NN.add(keras.layers.Dropout(0.2))
	# L2 regularization
	NN.add(keras.layers.Dense(3, kernel_regularizer='l2'))
	"""
	# Output layer
	NN.add(keras.layers.Dense(1, activation='sigmoid'))
	opt = keras.optimizers.Adam(learning_rate=0.0001)
	NN.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy',
                                          keras.metrics.Precision(name='Precision',thresholds=0.5),
                                          keras.metrics.Recall(name='Recall',thresholds=0.5)])
	return NN

def preprocess(file1, file0):
	# Read file1 and file 0 into a list of strings
	s1 = seqio.fasta2onehot(args.file1, 1)
	s2 = seqio.fasta2onehot(args.file0, 0)
	seqs = s1 + s2
	random.shuffle(seqs)  # scramble rows

	# Export list of strings into a temporary csv
	csv = "temp.csv"
	with open(csv, 'w') as fp:
		for item in seqs:
			fp.write(','.join(item))
			fp.write('\n')
	return csv

def train(file1, file0, epochs=20):
	input = preprocess(file1, file0)  # writes temp.csv to directory

	## Load data
	df = pd.read_csv("temp.csv")
	#df = df.sample(frac=1, random_state=1)  # scramble rows
	X = df.iloc[:,:-1]  # sequences
	Y = df.iloc[:,-1]  # labels
	(x_train,x_test,y_train,y_test) = train_test_split(X,Y,test_size=0.3, stratify=Y)

	## Training
	NN = create_model(x_train, 204)#, 10, 10, 10, 10, 10, 10, 10, 10, 10)
	result = NN.fit(x_train, y_train, epochs=epochs, verbose=1)

	## Write and format results
	f = open("results.txt", 'a')
	f.write("file1: " + file1[-10:-6] + '\n')
	f.write("file0: " + file0[-10:-6] + '\n')
	f.write(str(epochs) + " epochs\n")
	f.write("Results:\n")

	f.write("==========TRAINING==========\n")
	training_stats = NN.evaluate(x_train, y_train)
	print('Loss: ' + str(training_stats[0]))
	f.write('Loss: ' + str(training_stats[0]) + '\n')
	print('Training Accuracy: ' + str(training_stats[1]))
	f.write('Training Accuracy: ' + str(training_stats[1]) + '\n')
	f.write('Precision: ' + str(training_stats[2]) + '\n')
	f.write('Recall: ' + str(training_stats[3]) + '\n\n\n')
	#F1 = 2*((training_stats[2]*training_stats[3]) / (training_stats[2]+training_stats[3]))
	#f.write('F1: ' + str(F1) + '\n\n')

	f.write("==========TESTING==========\n")
	testing_stats = NN.evaluate(x_test, y_test)
	print('Loss: ' + str(testing_stats[0]))
	f.write('Loss: ' + str(testing_stats[0]) + '\n')
	print('Testing Accuracy: ' + str(testing_stats[1]))
	f.write('Testing Accuracy: ' + str(testing_stats[1]) + '\n')
	f.write('Testing Precision: ' + str(testing_stats[2]) + '\n')
	f.write('Testing Recall: ' + str(testing_stats[3]) + '\n\n\n')
	#F1 = 2*((testing_stats[2]*testing_stats[3]) / (testing_stats[2]+testing_stats[3]))
	#f.write('F1: ' + str(F1) + '\n\n\n\n')

	f.close()
	os.remove("temp.csv")
	return

## CLI
parser = argparse.ArgumentParser(description="Classify seqs as splice site or not")
parser.add_argument("--file1", required=True, type=str, metavar='<file>',
                    help="Enter name of fasta file of trues")
parser.add_argument("--file0", required=True, type=str, metavar='<file>',
                    help="Enter name of fasta file of falses")
parser.add_argument("--e", required=False, type=int, metavar='<file>',
                    help="Enter number of epochs")
#parser.add_argument("--output", required=True, type=str, metavar='<file>',
#                    help="Enter name of output file")
args = parser.parse_args()

if __name__ == "__main__":
    train(args.file1, args.file0, args.e)
