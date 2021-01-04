import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import sys




class DynamicNet(nn.Module):
	def __init__(self, input_dim, hidden_dim_array, non_linear_function_array, prob = []):
		super(DynamicNet, self).__init__()
		### for getattr
		self.input_dimension = input_dim
		self.hid_dim = hidden_dim_array
		self.nlfa = non_linear_function_array
		self.p = prob
		###

		self.linear_functions = []
		self.prob = prob
		self.non_linear_functions = [x() for x in non_linear_function_array]
		#print(self.non_linear_functions)

		self.hidden_layers = len(hidden_dim_array)
		for l in range(self.hidden_layers):
			assert(type(input_dim) == int)
			assert(type(hidden_dim_array[l]) == int)
			self.linear_functions.append(nn.Linear(input_dim, hidden_dim_array[l]))
			input_dim = hidden_dim_array[l]
		self.linear_functions = nn.ModuleList(self.linear_functions)
		self.final_layer = nn.Linear(input_dim, 1)

		### sanity checks
		if len(self.prob) != 0:
			assert (len(self.non_linear_functions) == len(self.prob) == len(self.linear_functions))
		else:
			assert (len(self.non_linear_functions) == len(self.linear_functions))


	def forward(self, x):
		out = x
		#out = out.flatten(start_dim = 1)
		for i in range(self.hidden_layers): #include the last layer as well
			out = self.linear_functions[i](out)
			#non_linear_function = self.non_linear_functions[i]
			out = self.non_linear_functions[i](out)
			if len(self.prob) != 0:
				assert(type(self.prob[i]) == float)
				assert(self.prob[i] <= 1.0)
				out = F.dropout(out, p=self.prob[i], training = True)
			###assert statement (sanity checks)
			#nn.functional.dropout(inputs, p=self.p, training=True)

			#t = F.dropout(t, training=self.training)
		out = torch.sigmoid(self.final_layer(out))
		return out

'''
Need:
input dimension
number of hidden layers
nodes in each of the hidden layers
activation energy
dropout
initialization
'''

