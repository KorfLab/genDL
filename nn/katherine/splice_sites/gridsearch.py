import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("cds0.csv")
df = df.sample(frac=1, random_state=1)  # scramble rows
x_train = df.iloc[:,:-1]
y_train = df.iloc[:,-1]

def create_model(layer1=5):
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

pick_layer_size(x_train, y_train)
