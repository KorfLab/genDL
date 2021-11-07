# NN for splice site classification

import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import argparse

def create_model(X, layer1=5):  # 5 is for testing
 NN = keras.Sequential()
  # Hidden layer 1
  NN.add(keras.layers.Dense(layer1, input_dim = X.shape[1], activation='sigmoid'))
  #Hidden layer 2
  #NN.add(keras.layers.Dense(5, activation='sigmoid'))
  # Output layer
  NN.add(keras.layers.Dense(1, activation='sigmoid'))
  NN.compile(loss='mse', optimizer='sgd', metrics=['accuracy',
                                          keras.metrics.Precision(name='Precision',thresholds=0.5),
                                          keras.metrics.Recall(name='Recall',thresholds=0.5)])
  return NN

# Pick layer size(s)
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
def pick_layer_size(x_train, y_train):
  model = KerasClassifier(build_fn=create_model)
  param_grid={'layer1':[3,6,9,12,15,18,21]}
  grid = GridSearchCV(estimator=model, param_grid=param_grid)
  grid_result = grid.fit(x_train, y_train)
  print((grid_result.best_score_, grid_result.best_params_))
  return

#pick_layer_size(x_train, y_train)

def train(input, eps):
  df = pd.read_csv(input)
  df = df.sample(frac=1, random_state=1)  # scramble rows
  X = df.iloc[:,:-1]  # sequences
  Y = df.iloc[:,-1]  # labels
  (x_train,x_test,y_train,y_test) = train_test_split(X,Y,test_size=0.3, stratify=Y)

  NN = create_model(x_train, 9)
  result = NN.fit(x_train, y_train, epochs=eps, verbose=1)

  f = open(input[:-4]+".results.txt", 'w')
  f.write(input[:-4] + " Results:\n")
  f.write(str(eps) + " epochs\n")

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

## CLI
parser = argparse.ArgumentParser(description="csv of trues and falses")
parser.add_argument("--input", required=True, type=str, metavar='<file>',
                    help="Enter name of input file")
parser.add_argument("--eps", required=True, type=int, metavar='<file>',
                    help="Enter name of input file")
#parser.add_argument("--output", required=True, type=str, metavar='<file>',
#                    help="Enter name of output file")
args = parser.parse_args()

if __name__ == "__main__":
    train(args.input, args.eps)
