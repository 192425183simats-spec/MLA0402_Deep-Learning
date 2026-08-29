import numpy as np
import matplotlib.pyplot as plt

# Sample data
X = np.array([1,2,3,4,5])
Y = np.array([2,4,6,8,10])

m = 0
b = 0

learning_rate = 0.01
iterations = 100

n = len(X)
losses = []

for i in range(iterations):

    y_pred = m*X + b

    loss = np.mean((Y-y_pred)**2)
    losses.append(loss)

    dm = (-2/n)*np.sum(X*(Y-y_pred))
    db = (-2/n)*np.sum(Y-y_pred)

    m = m-learning_rate*dm
    b = b-learning_rate*db

print("Slope =",m)
print("Intercept =",b)

plt.plot(losses)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Learning Curve")
plt.show()
