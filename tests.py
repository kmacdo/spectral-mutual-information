"""
Simple tests. Contains sample graphs
1. Swiss Roll
2. 
and sample distributions
1. 
"""
import numpy as np 
import matrix_method
np.random.seed(605)
X = np.random.rand(2,3)
Y = np.random.rand(5,4)
Y = 3*X
a = matrix_method.spectral_mutual_information(X,Y)
print(a)