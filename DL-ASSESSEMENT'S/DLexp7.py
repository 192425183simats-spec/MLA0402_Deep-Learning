from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron

iris = load_iris()

X_train,X_test,y_train,y_test = train_test_split(
iris.data,
iris.target,
test_size=0.2,
random_state=42)

mlp = MLPClassifier(hidden_layer_sizes=(20,),
max_iter=1000)

mlp.fit(X_train,y_train)

per = Perceptron()

per.fit(X_train,y_train)

print("MLP Accuracy:",
mlp.score(X_test,y_test))

print("Perceptron Accuracy:",
per.score(X_test,y_test))
