"""
Simple tests. Contains sample graphs
1. Swiss Roll
2. 
and sample distributions
1. 
"""
import numpy as np 
import matrix_method
import scipy.stats
np.random.seed(605)
X = np.random.rand(9,13)
# Y = np.random.rand(5,13)
Y = X
a = matrix_method.spectral_mutual_information(X,Y)
print(a)
