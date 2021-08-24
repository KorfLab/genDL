#!/usr/bin/python3

import argparse

import numpy as np
import torch
from torch import nn
from data_encoder import one_hot
import torch.nn.functional as F
from pandas import read_csv
from torch.utils.data import Dataset

path = 'data.csv' #fix hardcoding later

'''
def activation(x):
    return 1/(1+torch.exp(-x))
   
#def softmax(x):
#    return torch.exp(x)/torch.sum(torch.exp(x), dim=1).view(-1, 1)
'''




#need to have an encoded csv 
class read_CSV(Dataset): #why does this need to be a class?
	
	def __init__(self,path):
		df = read_csv(path, header=None) # load the csv file as a dataframe
		#print(df.columns)
		#print(df.values)
		
		self.X = df.values[:, :-1] # store the inputs 
		self.y = df.values[:,-1]  # and outputs
		#print(type(df))
	


titanic = read_CSV(path)

#print(titanic.X)
#print(titanic.y)

'''  
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(18, 20)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(20, 1)
        
        # Define sigmoid activation and softmax output 
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.F.relu(x)#????
        x = self.output(x)
        x = self.sigmoid(x)
        
        return x
        
model = Network()
print(model)


import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(784, 256)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(256, 10)
        
    def forward(self, x):
        # Hidden layer with sigmoid activation
        x = F.sigmoid(self.hidden(x))
        # Output layer with softmax activation
        x = F.softmax(self.output(x), dim=1)
        
        return x


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # Defining the layers, 128, 64, 10 units each
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        # Output layer, 10 units - one for each digit
        self.fc3 = nn.Linear(64, 10)
        
    def forward(self, x):
        """ Forward pass through the network, returns the output logits """
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.softmax(x) #dim =1
        
        return x

data = one_hot('train.csv')
#print(data)


model = Network()
print(model)

#for d in data:
	
print(model.fc1.weight)
print(model.fc1.bias)


data = one_hot('train.csv')

'''
"""
#write two (or three) functions: training and evaluation in one (can also be separate)
# need to write a test function (will not update models )

#need to convert to tensors and input to dataloader
idx = 0
ps = model.forward(data[idx])
#ps = model.forward(images[img_idx,:])
#data[0]
"""
