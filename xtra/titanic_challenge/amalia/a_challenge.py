import argparse
import sys
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder


parser = argparse.ArgumentParser(description='CNN model')
parser.add_argument('--file', required=True, type=str,
	metavar='<path>', help='path to cvs file')
parser.add_argument('--epoch', required=False, type=int, default = 5,
	metavar='<int>', help='number of epochs')
parser.add_argument('--batch', required=False, type=int, default = 1,
	metavar='<int>', help='batch size')
parser.add_argument('--seed', required=False, type=int,
	metavar='<int>', help='random seed')
arg = parser.parse_args()

if arg.seed:
	random.seed(arg.seed)

# load dataset
X = pd.read_csv(arg.file)
X['Embarked'] = X['Embarked'].astype(str)
#print(X.shape)
# limit to categorical data using df.select_dtypes()
#keeping, passengerID, Pclass, Embarked, Cabin, Fare, Age, Sex
#Label = X['Survived']
X = X.sample(frac = 1)
label = X['Survived']
X_train = X[['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch',
			 'Fare', 'Embarked']].copy()
#print(X_train.shape)

le = preprocessing.LabelEncoder()
#print(X_train.head())
X_2 = X_train.apply(le.fit_transform)
#print(X_2.head())


enc = preprocessing.OneHotEncoder()
enc.fit(X_2)

#Transform
onehotlabels = enc.transform(X_2).toarray()
print(onehotlabels.shape)

#label reformatting
label_2 = LabelEncoder().fit_transform(label)
label_2 = label_2.astype('float32')
label_2 = label_2.reshape((len(label_2), 1))
print(label_2.shape)

class MLP(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc1 = nn.Linear()
		self.fc2 = nn.Linear()
		self.fc2 = nn.Linear()






