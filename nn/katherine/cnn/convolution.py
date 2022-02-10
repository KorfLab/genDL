import numpy as np
import sys

# Using randomly generated 10x10 matrix to represent an image
#image = np.random.rand(10,10)
image = np.array([[0,1,2,3,4,5],
                  [0,1,2,3,4,5],
                  [0,1,2,3,4,5],
                  [0,1,2,3,4,5],
                  [0,1,2,3,4,5],
                  [0,1,2,3,4,5]])

# Filter size is chosen to be 3x3
# Filter values are what will be optimized
filter = np.random.rand(3,3)
print(filter)

m = image.shape[0]  # rows
n = image.shape[1]  # columns

for i in range(m-2):
  for j in range(n-2):
    submatrix = image[i:i+3, j:j+3]
    #prod = np.multiply(submatrix, filter).sum().sum()
    #print(prod)
    print(submatrix)
    print(np.amax(submatrix))
    sys.exit()
