import sklearn.datasets as datasets
import numpy as np
import optimize.general_gradient_descent as general_gradient_descent
from functools import partial
import matplotlib.pyplot as plt
import plot.decision_bound_plotter as bound_plotter
import function.impurity as impurity
from model.impurity.logistic_impurity import LogisticImpurity


'''
A= np.array([[1,2],[3,4]])
v = np.array([2,3])
prod = A*v[:,np.newaxis]
print("prod: ", prod)
'''


X, y = datasets.load_breast_cancer(return_X_y = True)
features = range(2)
set_size = X.shape[0]
X = X[:set_size,features]
y = y[:set_size]


model = LogisticImpurity()
model.train(X, y, 20000, .5)





'''
X_set,y = datasets.load_breast_cancer(return_X_y = True)
features = range(2)
X = np.ones((X_set.shape[0], len(features)+1))
for i in range(0, len(features)):
    X[:,i] = X_set[:,features[i]]

def f(X, params):
    right_probs = 1.0/(1+np.exp(-np.dot(X, params)))
    left_probs = 1-right_probs
    return np.asarray([left_probs, right_probs]).T

def dataset_continuous_impurity(params):
    return impurity.expected_gini(f(X, params), y)

params = np.random.rand(X.shape[1])*.001
params = general_gradient_descent.sgd_minimize(dataset_continuous_impurity, params, .1, 25000, .5)

def plot_predictor(X):
    assert(len(X.shape) == 1)
    X_func = np.ones(X.shape[0]+1)
    X_func[:X.shape[0]] = X
    probs = f(X_func, params)
    return np.argmax(probs)

ax = plt.gca()
bound_plotter.plot_contours(X[:,[0,1]], plot_predictor, ax, .025)
plt.scatter(X[:,0], X[:,1], color = ["blue" if y[i] == 0 else "red" for i in range(len(y))])
plt.show()
'''
