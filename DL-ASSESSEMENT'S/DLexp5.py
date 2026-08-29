import numpy as np

x = np.array([2,3])

weights = np.array([0.5,0.7])

bias = 1

z = np.dot(x,weights)+bias

sigmoid = 1/(1+np.exp(-z))

relu = max(0,z)

print("Weighted Sum =",z)

print("Sigmoid =",sigmoid)

print("ReLU =",relu)
