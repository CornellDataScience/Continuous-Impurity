import sklearn.datasets as datasets
import numpy as np
import optimize.general_gradient_descent as general_gradient_descent
from functools import partial
import matplotlib.pyplot as plt
import plot.decision_bound_plotter as bound_plotter
import function.impurity as impurity
import toolbox.data_helper as data_helper
from sklearn.datasets import fetch_mldata
from model.impurity.logistic_impurity_model import LogisticImpurityModel
from model.impurity.matrix_activation_logistic_impurity import MatrixActivationLogisticImpurity
from function.activation.sigmoid import Sigmoid
from function.activation.tanh import TanH
from function.activation.identity import Identity
#REMINDER: Use np.float_power instead of ** or np.power for fractional powers
#TODO: make a more general framework for treebased models using continuous impurity. I.e. make
#a more modular class that can be extended and have some abstract functions implemented to give
#the necessary functions for gradients, etc.
#TODO: make node in tree stop training when cost basically isnt' moving. (don't train the node if it's already like 99% accurate, etc.)
#TODO: fix NaN problems in expected gini, gradients, etc.
#TODO: Add more node train termination parameters to impurity tree
#TODO: Make a version of logistic trees that does not force being binary


X,y = datasets.load_breast_cancer(return_X_y = True)#datasets.load_iris(return_X_y = True)
X = X.astype(np.float64)
FEATURES = [0,1]
X = X[:,FEATURES]
print("x -: ",  np.average(X, axis = 0))
X -= np.average(X, axis = 0)

def change_basis(X):
    return X
    #return np.column_stack([X, X**2, X[:,0]*X[:,1]])

y[np.where(y==2)] = 1
#X -= np.mean(X[np.where(y==1)], axis = 0)
#X/=np.max(np.abs(X))
colors = [["blue", "red", "green"][y[i]] for i in range(y.shape[0])]
plt.scatter(X[:,0], X[:,1], color = colors)
plt.show()

'''
mnist = fetch_mldata('MNIST original')
X = mnist.data.astype('float64')
y = mnist.target
'''

print("X max: ", X.max())
print("X min: ", X.min())
NUM_TRAIN = int(0.8*X.shape[0])
#np.random.seed(seed = 42)
PERMUTE_INDS = np.random.permutation(np.arange(0, X.shape[0]))

TRAIN_INDS = PERMUTE_INDS[:NUM_TRAIN]
TEST_INDS = PERMUTE_INDS[NUM_TRAIN:]
X_train = change_basis(X[TRAIN_INDS])
y_train = y[TRAIN_INDS]
X_test = change_basis(X[TEST_INDS])
y_test = y[TEST_INDS]

plt.show()
'''
PROBLEM: using T(Ax) may not be nonlinearenough to represent any interesting transformation. Even though
the idea is to nest them using a tree-like structure, often makes all the axises look practically equal (points generally lie on y = x plot)
'''
model = MatrixActivationLogisticImpurity(TanH(), X_train.shape[1], 4)#LogisticImpurityModel(X_train.shape[1])#
model.train(X_train, y_train, 10000, [1,1], print_progress_iters = 1000)


predictions = model.predict(X_test)
num_right = np.sum(predictions==y_test)
print("Accuracy: ", float(100.0 * num_right/float(y_test.shape[0])))


def pred_func(X):
    print("x shape; ", X.shape)
    #print("outs: ", np.argmax(model.predict(X), axis = 1))
    #return np.argmax(model.predict(X), axis = 1)
    X = change_basis(X)
    out = np.zeros(X.shape[0])
    preds = model.predict(X)
    out[np.where(preds[:,0] > 0.5)] = 1
    return out

ax = plt.gca()
bound_plotter.plot_contours(X, pred_func, ax, .025)
colors = [["blue", "red", "green"][y[i]] for i in range(y.shape[0])]
plt.scatter(X[:,0], X[:,1], color = colors)
plt.show()


X_transformed = model._get_mat_act_transform().transform(data_helper.affine_X(X))
plt.scatter(X_transformed[:,0], X_transformed[:,1], color = colors)
plt.show()



'''
X, y = datasets.load_iris(return_X_y = True)
features = [2,3]
set_size = X.shape[0]
X = X[:set_size,features]
y = y[:set_size]
y[np.where(y == 2)] = 0

#When training impurity trees, ensure training params are good enough that
#models won't have to stop training before making a good split (splitting when
#one subset is empty or almost empty will cause the tree to terminate before
#reaching a good depth)

model = LogisticImpurityTree()
model.train(X, y, 5, 10, 50000, .05)

ax = plt.gca()
bound_plotter.plot_contours(X, model.predict, ax, .005)
colors = [["blue", "red", "green"][y[i]] for i in range(y.shape[0])]
plt.scatter(X[:,0], X[:,1], color = colors)
plt.show()
'''
