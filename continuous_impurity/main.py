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
import model.impurity.greedy_impurity_tree_builder as greedy_impurity_tree_builder
from model.impurity.global_impurity_model_tree import GlobalImpurityModelTree, Node, NodeModel

#REMINDER: Use np.float_power instead of ** or np.power for fractional powers
#TODO: make a more general framework for treebased models using continuous impurity. I.e. make
#a more modular class that can be extended and have some abstract functions implemented to give
#the necessary functions for gradients, etc.
#TODO: make node in tree stop training when cost basically isnt' moving. (don't train the node if it's already like 99% accurate, etc.)
#TODO: fix NaN problems in expected gini, gradients, etc.
#TODO: Add more node train termination parameters to impurity tree
#TODO: Make a version of logistic trees that does not force being binary


X,y = datasets.load_breast_cancer(return_X_y = True)#datasets.load_iris(return_X_y = True)#
X = X.astype(np.float64)
FEATURES =[0,1]#[0,1]#
X = X[:,FEATURES]


X = data_helper.unit_square_normalize(X)
X = data_helper.mean_center(X)

print("X maxes: ", np.max(X, axis = 0))
print("X mins: ", np.min(X, axis = 0))

#y[np.where(y==2)] = 1
#X -= np.mean(X[np.where(y==1)], axis = 0)
#X/=np.max(np.abs(X))

colors = [["blue", "red", "green"][y[i]] for i in range(y.shape[0])]
plt.scatter(X[:,0], X[:,1], color = colors)
plt.show()


NUM_TRAIN = int(0.8*X.shape[0])
#np.random.seed(seed = 42)
PERMUTE_INDS = np.random.permutation(np.arange(0, X.shape[0]))

TRAIN_INDS = PERMUTE_INDS[:NUM_TRAIN]
TEST_INDS = PERMUTE_INDS[NUM_TRAIN:]
X_train = X[TRAIN_INDS]
y_train = y[TRAIN_INDS]
X_test = X[TEST_INDS]
y_test = y[TEST_INDS]

plt.show()

#model = greedy_impurity_tree_builder.build_logistic_impurity_tree(X_train.shape[1], 2)
#model = greedy_impurity_tree_builder.build_matrix_activation_logistic_impurity_tree([TanH(), TanH(), TanH()], [4,4,4], X.shape[1])#
#model.train(None, X,y,10,20000,[1,1])

def model_fn(params, k,  X):
    X_affine = data_helper.affine_X(X)
    k_eq_0_out = 1.0/(1+np.exp(-np.dot(X_affine, params)))
    return k_eq_0_out if k==0 else 1-k_eq_0_out

def grad_model_fn(params, k, X):
    X_affine = data_helper.affine_X(X)
    k_eq_0_func_out = model_fn(params, 0, X)
    k_eq_0_out = (k_eq_0_func_out*(1-k_eq_0_func_out))*X_affine
    return k_eq_0_out if k == 0 else -k_eq_0_out

def create_logistic_regression_node_model(x_shape):
    params = .0000001 * (np.random.rand((x_shape + 1)) - .5).astype(np.float64)
    return NodeModel(params, model_fn, grad_model_fn)


model_head = Node(None, create_logistic_regression_node_model(X.shape[1]))
head_child1 = Node(model_head, create_logistic_regression_node_model(X.shape[1]))
head_child2 = Node(model_head, None)
model_head.add_child(head_child1)
model_head.add_child(head_child2)
head_child11 = Node(head_child1, None)
head_child12 = Node(head_child1, None)
head_child1.add_child(head_child11)
head_child1.add_child(head_child12)
model = GlobalImpurityModelTree(model_head)
model.train(X_train, y_train, 2500, .01)


predictions = model.predict(X_test)
num_right = np.sum(predictions==y_test)
print("Accuracy: ", float(100.0 * num_right/float(y_test.shape[0])))


def pred_func(X):
    return model.predict(X)


ax = plt.gca()
bound_plotter.plot_contours(X, pred_func, ax, .05)
colors = [["blue", "red", "green"][y[i]] for i in range(y.shape[0])]
plt.scatter(X[:,0], X[:,1], color = colors)
plt.show()


X_transformed = model._get_mat_act_transform().transform(data_helper.affine_X(X))
plt.scatter(X_transformed[:,0], X_transformed[:,1], color = colors)
plt.show()
