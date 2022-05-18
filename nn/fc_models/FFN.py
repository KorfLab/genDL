import argparse
import gzip
import numpy as np
import os  # for removing temp file later
import pandas as pd
import random  # to shuffle rows
import seqio
from sklearn.model_selection import train_test_split
from tensorflow import keras

## CLI
parser = argparse.ArgumentParser(description="FFN for small motifs")
parser.add_argument("--file1", required=True, type=str, metavar="<file>",
                    help="txt file of trues")
parser.add_argument("--file0", required=True, type=str, metavar="<file>",
                    help="txt file of falses")
parser.add_argument("--e", required=False, type=int, metavar="<int>",
                    default=20, help="number of epochs")
parser.add_argument("--fasta", action="store_true")
parser.add_argument("--hidden", action="store_true")
args = parser.parse_args()

def encode(infile):
    d = {'A':[1,0,0,0],
        'C':[0,1,0,0],
        'G':[0,0,1,0],
        'T':[0,0,0,1]}
    if infile.endswith(".gz"):
        f = gzip.open(infile, "rt")
    else:
        f = open(infile, "r")
    X = []
    while True:
        line = f.readline()
        line = line.strip()
        if len(line) == 0:
            break
        seq = []
        for nt in line:
            seq.extend(d[nt])
        X.append(seq)
    X = np.array(X)
    return X

def load_fasta(file1, file0):
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

if __name__ == "__main__":
	# Load data and do one-hot encoding
	if args.fasta:
		input = load_fasta(args.file1, args.file0)  # writes temp.csv to directory
		## Load data
		df = pd.read_csv("temp.csv")
		#df = df.sample(frac=1, random_state=1)  # scramble rows
		X = df.iloc[:,:-1]  # sequences
		type(X)
		Y = df.iloc[:,-1]  # labels
	else:
		f1 = encode(args.file1)
		f0 = encode(args.file0)
		# Concatenate f1 and f0
		X = np.concatenate((f1, f0), axis=0)
		Y = np.concatenate((np.ones(len(f0)), np.zeros(len(f0))), axis=None)  #labels

	# Train/test splitting and shuffle rows
	x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

	## Model
	NN = keras.Sequential()

    # Hidden layer 1
	layer1 = keras.layers.Dense(9, input_dim=X.shape[1], activation="sigmoid")
	NN.add(layer1)
	NN.add(keras.layers.Dropout(0.2))
	NN.add(keras.layers.Dense(3, kernel_regularizer="l2"))

    # Hidden layer 2
    #NN.add(keras.layers.Dense(50, input_dim=X.shape[1], activation="sigmoid"))
    #NN.add(keras.layers.Dropout(0.2))
    #NN.add(keras.layers.Dense(3, kernel_regularizer="l2"))

    # Output layer
	output_layer = keras.layers.Dense(1, activation="sigmoid")
	NN.add(output_layer)

    # Compile
	NN.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])

    ## Train
	result = NN.fit(x_train, y_train, batch_size=32, epochs=args.e, verbose=1)

    ## Evaluate
    #training_stats = NN.evaluate(x_train, x_test)
    #print("Training Accuracy:", training_stats[1])
	testing_stats = NN.evaluate(x_test, y_test)
	print("Testing Accuracy:", testing_stats[1])

	if args.hidden:
		weights = output_layer.get_weights()
	else:
		weights = layer1.get_weights()
	#print(len(weights))
	#print(len(weights[0]))
    #print(weights[0]) # weights of input layer -> hidden
    #print(weights[1]) # weights of hidden layer -> output

	## Finish up
	if args.fasta:
		os.remove("temp.csv")
	if args.hidden:
		np.savetxt("../output/"+args.file1[:-7]+'.weights.hidden.csv', weights[0], delimiter=' ', fmt='%s')
	else:
		np.savetxt("../output/"+args.file1[:-7]+'.weights.csv', weights[0], delimiter=' ', fmt='%s')
