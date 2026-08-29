from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

X,y = make_regression(n_samples=1000,
n_features=1,
noise=20)

sgd = SGDRegressor(max_iter=1000)

sgd.fit(X,y)

batch = LinearRegression()

batch.fit(X,y)

print("SGD Score =",sgd.score(X,y))

print("Batch Score =",batch.score(X,y))
