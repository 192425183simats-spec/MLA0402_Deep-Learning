from sklearn.linear_model import Perceptron
from sklearn.datasets import make_classification
from sklearn.datasets import make_circles

# Linearly separable
X1,y1 = make_classification(n_samples=200,
n_features=2,
n_redundant=0,
n_clusters_per_class=1)

model = Perceptron()

model.fit(X1,y1)

print("Linear Accuracy:",
model.score(X1,y1))

# Non-linear
X2,y2 = make_circles(200,
noise=0.1,
factor=0.5)

model.fit(X2,y2)

print("Non-linear Accuracy:",
model.score(X2,y2))
