import numpy as np

learning_rate = 0.1

weight = 0.5

x = 2

target = 1

prediction = weight*x

error = prediction-target

gradient = error*x

new_weight = weight-learning_rate*gradient

print("Old Weight =",weight)

print("New Weight =",new_weight)
