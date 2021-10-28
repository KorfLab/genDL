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

def create_model(X, layer1=5, layer2=5, layer3=5, layer4=5):  # Using 5 as a placeholder
  NN = keras.Sequential()
  # Hidden layer 1
  NN.add(keras.layers.Dense(layer1, input_dim = X.shape[1], activation='sigmoid'))
  # Hidden layer 2
  NN.add(keras.layers.Dense(layer2, activation='sigmoid'))
  # Hidden layer 3
  NN.add(keras.layers.Dense(layer3, activation='sigmoid'))
  # Hidden layer 4
  NN.add(keras.layers.Dense(layer4, activation='sigmoid'))
  # Output layer
  NN.add(keras.layers.Dense(1, activation='sigmoid'))
  NN.compile(loss='mse', optimizer='adam', metrics=['accuracy',
                                          keras.metrics.Precision(name='Precision',thresholds=0.5),
                                          keras.metrics.Recall(name='Recall',thresholds=0.5)])
  return NN

def preprocess(file1, file0):
	# Read file1 and file 0 into a list of strings
	s1 = seqio.fasta2binary(args.file1, 1)
	s2 = seqio.fasta2binary(args.file0, 0)
	seqs = s1 + s2
	random.shuffle(seqs)  # scramble rows

	# Export list of strings into a temporary csv
	csv = "temp.csv"
	with open(csv, 'w') as fp:
		for item in seqs:
			fp.write(','.join(item))
			fp.write('\n')
	return csv

def train(file1, file0, epochs):
	input = preprocess(file1, file0)  # writes temp.csv to directory
	df = pd.read_csv("temp.csv")
  #df = df.sample(frac=1, random_state=1)  # scramble rows
	X = df.iloc[:,:-1]  # sequences
	Y = df.iloc[:,-1]  # labels
	(x_train,x_test,y_train,y_test) = train_test_split(X,Y,test_size=0.3, stratify=Y)

	NN = create_model(x_train, 15, 10, 5)
	result = NN.fit(x_train, y_train, epochs=epochs, verbose=1)

	## Write and format results
	f = open(file1[:-6] + ".results.txt", 'w')
	f.write("file1: " + file1[-10:-6] + '\n')
	f.write("file0: " + file0[-10:-6] + '\n')
	f.write("Results:\n")
	f.write(str(epochs) + " epochs\n")

	f.write("==========TRAINING==========\n")
	training_stats = NN.evaluate(x_train, y_train)
	print('Loss: ' + str(training_stats[0]))
	f.write('Loss: ' + str(training_stats[0]) + '\n')
	print('Training Accuracy: ' + str(training_stats[1]))
	f.write('Training Accuracy: ' + str(training_stats[1]) + '\n')
	print('Precision: ' + str(training_stats[2]))
	f.write('Precision: ' + str(training_stats[2]) + '\n')
	print('Recall: ' + str(training_stats[3]))
	f.write('Recall: ' + str(training_stats[3]) + '\n\n')

	f.write("==========TESTING==========\n")
	testing_stats = NN.evaluate(x_test, y_test)
	print('Loss: ' + str(testing_stats[0]))
	f.write('Loss: ' + str(testing_stats[0]) + '\n')
	print('Testing Accuracy: ' + str(testing_stats[1]))
	f.write('Testing Accuracy: ' + str(testing_stats[1]) + '\n')
	print('Precision: ' + str(testing_stats[2]))
	f.write('Testing Precision: ' + str(testing_stats[2]) + '\n')
	print('Recall: ' + str(testing_stats[3]))
	f.write('Testing Recall: ' + str(testing_stats[3]) + '\n')

	f.close()

	os.remove("temp.csv")
	return

## CLI
parser = argparse.ArgumentParser(description="Classify seqs as splice site or not")
parser.add_argument("--file1", required=True, type=str, metavar='<file>',
                    help="Enter name of fasta file of trues")
parser.add_argument("--file0", required=True, type=str, metavar='<file>',
                    help="Enter name of fasta file of falses")
parser.add_argument("--e", required=True, type=int, metavar='<file>',
                    help="Enter number of epochs")
#parser.add_argument("--output", required=True, type=str, metavar='<file>',
#                    help="Enter name of output file")
args = parser.parse_args()

if __name__ == "__main__":
    train(args.file1, args.file0, args.e)
    #pick_layer_size(args.input)
