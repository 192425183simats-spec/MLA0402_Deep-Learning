import numpy as np
import matplotlib.pyplot as plt

x = 8

lr = 0.1

history=[]

for i in range(50):

    grad = 2*x

    x = x-lr*grad

    history.append(x*x)

plt.plot(history)

plt.xlabel("Iterations")

plt.ylabel("Cost")

plt.show()
