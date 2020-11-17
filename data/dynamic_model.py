import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import sys




class DynamicNet(nn.Module):
    def __init__(self, input_dim, hidden_dim_array, non_linear_function_array):
        super().__init__()
        self.linear_functions = []
        #self.non_linear_functions = non_linear_function_array
        #self.non_linear_function = []
        #print(self.non_linear_functions)

        self.non_linear_functions = [x() for x in non_linear_function_array]
        #print(self.non_linear_functions)

        self.hidden_layers = len(hidden_dim_array)
        for l in range(self.hidden_layers):
            self.linear_functions.append(nn.Linear(input_dim, hidden_dim_array[l]))
            input_dim = hidden_dim_array[l]
        self.linear_functions = nn.ModuleList(self.linear_functions)
        self.final_layer = nn.Linear(input_dim, 1)

    def forward(self, x):
        out = x
        out = out.flatten(start_dim = 1)
        for i in range(self.hidden_layers): #include the last layer as well
            #t = F.elu(self.fc1(t))
            out = self.linear_functions[i](out)
            #print(out)
            #print(len(out))
            #non_linear_function = self.non_linear_functions[i]
            out = self.non_linear_functions[i](out)
            out = F.dropout(out, p=0.2, training = True)
            ##make it optional as well as customatize p
            ###assert statement (sanity checks)
            #nn.functional.dropout(inputs, p=self.p, training=True)

            #t = F.dropout(t, training=self.training)
        #out = self.final_linear(out)
        ###might what to use other functions (specify the last layer)
        out = torch.sigmoid(self.final_layer(out))
        #out = self.final_layer(x)
        #t = torch.sigmoid(self.out(t))
        return out
