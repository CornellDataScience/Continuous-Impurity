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
from model.impurity.global_impurity.global_impurity_node2 import GlobalImpurityNode2
from model.impurity.global_impurity.node_model2 import NodeModel2
from model.impurity.global_impurity.global_impurity_model_tree2 import GlobalImpurityModelTree2

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
FEATURES = [0,1]#range(X.shape[1])#
X = X[:,FEATURES]


X = data_helper.unit_square_normalize(X)

X = data_helper.mean_center(X)
X = data_helper.affine_X(X)
print("X maxes: ", np.max(X, axis = 0))
print("X mins: ", np.min(X, axis = 0))

#y[np.where(y==2)] = 1
#X -= np.mean(X[np.where(y==1)], axis = 0)
#X/=np.max(np.abs(X))

colors = [["blue", "red", "green"][y[i]] for i in range(y.shape[0])]
plt.scatter(X[:,0], X[:,1], color = colors)
plt.show()


NUM_TRAIN = int(.5*X.shape[0])
#np.random.seed(seed = 42)
PERMUTE_INDS = np.random.permutation(np.arange(0, X.shape[0]))

TRAIN_INDS = PERMUTE_INDS[:NUM_TRAIN]
TEST_INDS = PERMUTE_INDS[NUM_TRAIN:]
X_train = X[TRAIN_INDS]
y_train = y[TRAIN_INDS]
X_test = X[TEST_INDS]
y_test = y[TEST_INDS]

plt.show()

def create_dud_node_model2():
    def dud_func(params_dict, k, X):
        k_eq_0_out = 1.0/(1.0 + np.exp(-np.dot(X, params_dict["k1"])))
        return k_eq_0_out if k == 0 else 1-k_eq_0_out
        '''
        if k == 0:
            return (np.full(X.shape[0], 0.75))
        return (np.full(X.shape[0], 0.25))
        '''

    def dud_grad_func(params_dict, k, X):
        k_eq_0_out = dud_func(params_dict, 0, X)
        grad_k_eq_0_out = (k_eq_0_out*(1-k_eq_0_out))[:,np.newaxis] * X
        return {"k1":grad_k_eq_0_out} if k == 0 else {"k1":-grad_k_eq_0_out}
        '''
        out = {}
        for key in params_dict:
            out[key] = np.zeros((X.shape[0],) + params_dict[key].shape, dtype = params_dict[key].dtype)
        return out
        '''

    params_dict = {"k1": 0.000001*(np.random.rand((X.shape[1]))*0.5)}
    return NodeModel2(params_dict, dud_func, dud_grad_func)

def create_dud_tree(depth):
    def build(node, remaining_depth):
        if remaining_depth == 0:
            return None
        for i in range(0, 2):
            add_child = GlobalImpurityNode2(node, create_dud_node_model2()) \
                if remaining_depth != 1 else \
                GlobalImpurityNode2(node, None)
            node.add_children(add_child)
        for child in node._children:
            build(child, remaining_depth - 1)


    head = GlobalImpurityNode2(None, create_dud_node_model2())
    build(head, depth)
    return head

head = create_dud_tree(5)


'''
TODO: Move static methods in GlobalImpurityNode2 to GlobalImpurityModelTree2,
Check correctness of these methods in GlobalImpurityModelTree2, (pretty sure should be monotone decreasing,
but saw cost increase despite not nearing the optimum quite yet)
Speed these methods up,
make the func and grad_func more stable (was getting NaNs during training after
what appeared to be fairly good convergence)
'''
model = GlobalImpurityModelTree2(head)
model.train( X, y, 100000, 1)












'''
def change_basis(X):
    return data_helper.affine_X(X)


def model_fn(params, k,  X):
    X_hat = change_basis(X)
    k_eq_0_out = 1.0/(1+np.exp(-np.dot(X_hat, params)))
    return k_eq_0_out if k==0 else 1-k_eq_0_out

def grad_model_fn(params, k, X):
    X_hat = change_basis(X)
    k_eq_0_func_out = model_fn(params, 0, X)
    k_eq_0_out = (k_eq_0_func_out*(1-k_eq_0_func_out))[:,np.newaxis]*X_hat
    #to dampen the bias gradient so does not overtake other params
    #k_eq_0_out[:, k_eq_0_out.shape[1] -1] *= 0.0001
    return k_eq_0_out if k == 0 else -k_eq_0_out

def create_logistic_regression_node_model(X):
    x_shape = change_basis(X).shape[1]
    params = .0000001 * (np.random.rand((x_shape)) - .5).astype(np.float64)
    return NodeModel(params, model_fn, grad_model_fn)

def construct_tree(X, depth, model_func):
    def f(node, depth_remaining):
        if depth_remaining == 0:
            child1 = Node(node, None)
            child2 = Node(node, None)
            node.add_children([child1, child2])
            return None

        child1 = Node(node, model_func(X))
        child2 = Node(node, model_func(X))
        node.add_children([child1, child2])
        for child in node._children:
            f(child, depth_remaining - 1)
    head = Node(None, model_func(X))
    f(head, depth)
    return head


model_head = construct_tree(X, 3, create_logistic_regression_node_model)
model = GlobalImpurityModelTree(model_head)
model.train(X_train, y_train, 5000, 10)


predictions = model.predict(X)
num_right = np.sum(predictions==y)
print("Accuracy: ", float(100.0 * num_right/float(y.shape[0])))


def pred_func(X):
    return model.predict(X)


ax = plt.gca()
bound_plotter.plot_contours(X, pred_func, ax, .0025)
colors = [["blue", "red", "green"][y[i]] for i in range(y.shape[0])]
plt.scatter(X[:,0], X[:,1], color = colors)
plt.show()
'''
