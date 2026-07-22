import numpy as np

inputs = np.array([1,2])

weights = np.array([[0.2,0.4],
                    [0.5,0.3]])

bias = np.array([0.1,0.2])

hidden = np.dot(inputs,weights)+bias

output = 1/(1+np.exp(-hidden))

print("Hidden Layer Output")

print(output)
