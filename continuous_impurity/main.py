import sklearn.datasets as datasets
import numpy as np
import optimize.general_gradient_descent as general_gradient_descent
from functools import partial
import matplotlib.pyplot as plt
import plot.decision_bound_plotter as bound_plotter
import function.impurity as impurity
from model.impurity.logistic_impurity import LogisticImpurity
from model.impurity.logistic_impurity_tree import LogisticImpurityTree
import toolbox.data_helper as data_helper
from model.impurity.impurity_model import ImpurityModel

#TODO: make a more general framework for treebased models using continuous impurity. I.e. make
#a more modular class that can be extended and have some abstract functions implemented to give
#the necessary functions for gradients, etc.

X, y = datasets.load_iris(return_X_y = True)
features = [2,3]
set_size = X.shape[0]
X = X[:set_size,features]
y = y[:set_size]

#When training impurity trees, ensure training params are good enough that
#models won't have to stop training before making a good split (splitting when
#one subset is empty or almost empty will cause the tree to terminate before
#reaching a good depth)
'''
model = LogisticImpurityTree()
model.train(X, y, 5, 10, 50000, .05)
'''

class LogisticModel:


    def rand_init_params(self, X):
        self.__theta = np.random.rand(X.shape[1]+1)*.001

    def func(self, X):
        X_affine = data_helper.affine_X(X)
        left_probs = 1.0/(1.0+np.exp(-np.dot(X_affine, self.__theta)))
        return np.asarray([left_probs, 1-left_probs]).T

    def d_func(self, x, k, p, j):
        x_affine = np.ones(x.shape[0]+1)
        x_affine[:x.shape[0]] = x
        outs = self.func(np.asarray([x]))[0]
        left_d_func = x_affine[j]*outs[k]*(1-outs[k])
        return left_d_func if k == 0 else -left_d_func

    def step_params(self, step):
        self.__theta += step[0]

    def params_shape(self):
        return (2, self.__theta.shape[0])

model = ImpurityModel(LogisticModel())
model.train(X,y,50000,.5)

ax = plt.gca()
bound_plotter.plot_contours(X, model.predict, ax, .005)
colors = [["blue", "red", "green"][y[i]] for i in range(y.shape[0])]
plt.scatter(X[:,0], X[:,1], color = colors)
plt.show()





'''
X, y = datasets.load_iris(return_X_y = True)
features = [1,3]
set_size = X.shape[0]
X = X[:set_size,features]
y = y[:set_size]


model = LogisticImpurity()
model.train(X, y, 20000, .05)


def plot_predictor(X):
    out = np.zeros(X.shape[0])
    predicts = model.predict(X)
    out[np.where(predicts>.5)] = 1
    return out

ax = plt.gca()
bound_plotter.plot_contours(X, plot_predictor, ax, .025)
colors = [["blue", "red", "green"][y[i]] for i in range(y.shape[0])]
plt.scatter(X[:,0], X[:,1], color = colors)
plt.show()
'''
