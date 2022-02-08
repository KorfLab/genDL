import argparse
import gzip
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split

## CLI
parser = argparse.ArgumentParser(description="CNN for genomic data")
parser.add_argument("--file1", required=True, type=str, metavar='<file>',
                    help="txt file of trues")
parser.add_argument("--file0", required=True, type=str, metavar='<file>',
                    help="txt file of falses")
parser.add_argument("--e", required=False, type=int, metavar='<int>',
                    help="Enter number of epochs")
args = parser.parse_args()

# For one-hot encoding
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
        line = line.strip() # strip newline character
        if len(line) == 0:
            break
        seq = []
        for nt in line:
            seq.append(d[nt])
        X.append(seq)
    X = np.array(X)
    return X

if __name__ == "__main__":
    # One-hot encoding
    f1 = encode(args.file1)
    f0 = encode(args.file0)
    # Concatenate f1 and f0
    X = np.concatenate((f1, f0), axis=0)
    Y = np.concatenate((np.ones(len(f0)), np.zeros(len(f0))), axis=None)  # labels
    # Train/test splitting
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    # Reshape inputs
    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)

    ## Construct the model
    CNN = Sequential()

    # Conv 1
    CNN.add(Conv2D(64, kernel_size=(5,1), activation="relu", input_shape=x_train[0].shape)) # try other powers of 2?
    #CNN.add(Dropout(0.50))
    CNN.add(MaxPooling2D(pool_size=(2,1)))

    # Conv 2
    CNN.add(Conv2D(50, (4,2), activation='relu', kernel_regularizer=keras.regularizers.l2(1e-2)))
    #CNN.add(Dropout(0.50))
    CNN.add(MaxPooling2D((2, 1)))

    # Conv 3
    CNN.add(Conv2D(15, (4,3), activation='relu', kernel_regularizer=keras.regularizers.l2(1e-2)))
    #CNN.add(Dropout(0.50))
    CNN.add(MaxPooling2D((2, 1)))

    CNN.add(Flatten()) # need to flatten input so it can go through dense layer
    CNN.add(Dense(1, activation="sigmoid"))

    ## Compile
    CNN.compile(loss="binary_crossentropy", # categorical_crossentropy
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            metrics=["accuracy"])

    ## Train
    CNN.fit(x_train, y_train, epochs=16, batch_size=256, shuffle=True)

    ## Evaluate
    testing_stats = CNN.evaluate(x_test, y_test)
    print('Testing Accuracy: ' + str(testing_stats[1]))
