import argparse
import gzip
import numpy as np
import os  # for removing temp file later
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

if __name__ == "__main__":
    # Load data and do one-hot encoding
    f1 = encode(args.file1)
    f0 = encode(args.file0)
    # Concatenate f1 and f0
    X = np.concatenate((f1, f0), axis=0)
    Y = np.concatenate((np.ones(len(f0)),
                        np.zeros(len(f0))), axis=None)  #labels
    # Train/test splitting and shuffle rows
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    ## Model
    NN = keras.Sequential()

    # Hidden layer 1
    NN.add(keras.layers.Dense(9, input_dim=X.shape[1], activation="sigmoid"))
    NN.add(keras.layers.Dropout(0.2))
    NN.add(keras.layers.Dense(3, kernel_regularizer="l2"))

    # Hidden layer 2
    #NN.add(keras.layers.Dense(50, input_dim=X.shape[1], activation="sigmoid"))
    #NN.add(keras.layers.Dropout(0.2))
    #NN.add(keras.layers.Dense(3, kernel_regularizer="l2"))

    # Output layer
    NN.add(keras.layers.Dense(1, activation="sigmoid"))

    # Compile
    NN.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])

    ## Train
    result = NN.fit(x_train, y_train, batch_size=32, epochs=args.e, verbose=1)

    ## Evaluate
    #training_stats = NN.evaluate(x_train, x_test)
    #print("Training Accuracy:", training_stats[1])
    testing_stats = NN.evaluate(x_test, y_test)
    print("Testing Accuracy:", testing_stats[1])
