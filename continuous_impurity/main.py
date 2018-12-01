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
from function.activation.relu import Relu
from function.activation.identity import Identity
import model.impurity.greedy_impurity_tree_builder as greedy_impurity_tree_builder
from model.impurity.global_impurity.global_impurity_node2 import GlobalImpurityNode2
from model.impurity.global_impurity.node_model2 import NodeModel2
from model.impurity.global_impurity.global_impurity_model_tree2 import GlobalImpurityModelTree2
import model.impurity.global_impurity.global_impurity_tree_maker2 as global_impurity_tree_maker2
import toolbox.data_maker as data_maker

from sklearn import tree
import math
#REMINDER: Use np.float_power instead of ** or np.power for fractional powers
#TODO: make a more general framework for treebased models using continuous impurity. I.e. make
#a more modular class that can be extended and have some abstract functions implemented to give
#the necessary functions for gradients, etc.
#TODO: make node in tree stop training when cost basically isnt' moving. (don't train the node if it's already like 99% accurate, etc.)
#TODO: fix NaN problems in expected gini, gradients, etc.
#TODO: Add more node train termination parameters to impurity tree
#TODO: Make a version of logistic trees that does not force being binary


#X,y =  datasets.make_classification(n_samples = 200, n_features=2, n_redundant=0, n_informative=2,random_state=2, n_clusters_per_class=2)
#X,y = datasets.load_digits(return_X_y = True)##datasets.load_iris(return_X_y = True)#datasets.load_breast_cancer(return_X_y = True)#datasets.make_moons()#
X, y = data_maker.create_rect_simple()
X = X.astype(np.float64)

ROT = math.pi/4.0
ROT_MAT = np.array([[np.cos(ROT), -np.sin(ROT)],[np.sin(ROT), np.cos(ROT)]])

X = np.dot(ROT_MAT, X.T).T
FEATURES = range(X.shape[1])#[0,1]#
X = X[:,FEATURES]


X = data_helper.unit_square_normalize(X)
#X/=16
X = data_helper.mean_center(X)

print("X maxes: ", np.max(X, axis = 0))
print("X mins: ", np.min(X, axis = 0))

#y[np.where(y==2)] = 1
#X -= np.mean(X[np.where(y==1)], axis = 0)
#X/=np.max(np.abs(X))

if X.shape[1] == 2:
    colors = [["blue", "red", "green"][y[i]] for i in range(y.shape[0])]
    plt.scatter(X[:,0], X[:,1], color = colors)
    plt.show()





NUM_TRAIN = int(0.9*X.shape[0])
#np.random.seed(seed = 42)
PERMUTE_INDS = np.random.permutation(np.arange(0, X.shape[0]))

TRAIN_INDS = PERMUTE_INDS[:NUM_TRAIN]
TEST_INDS = PERMUTE_INDS[NUM_TRAIN:]
X_train = X[TRAIN_INDS]
y_train = y[TRAIN_INDS]
X_test = X[TEST_INDS]
y_test = y[TEST_INDS]




'''
TODO:
make the func and grad_func more stable (was getting NaNs during training after
what appeared to be fairly good convergence)

Experiment with using a gaussian distribution as the split function... see
if it can make some kind of interesting generative classifier

TODO: add a function that gives a node leaf number each x falls into for visualization's sake

ISSUE: globally optimized imprity trees may limit themselves by having certain nodes be responsible for a split
    despite not having enough remaining children to accuratley classify the remainder? Fix would be to not make the
    tree of fixed depth and to grow it dynamically as it trains (which should even be possible?). Would just have to have
    each train step possibly even alter the structure of the tree, i.e. chopping off children if a node is good enough to
    act as a leaf, or adding children if a leaf is not good enough to act as a leaf.
'''

head = global_impurity_tree_maker2.construct_logistic_tree(5, X.shape[1])#global_impurity_tree_maker2.construct_matrix_activation_logistic_tree(X.shape[1], TanH(), [3,3,3])#

'''
TODO: Experiment with fact that tree may naturally provide "more confident" regions (see leaf predicts vs. class predicts).
    Basically, certain leaves often have almost only one class fall into them. These are "very confident" classification regions
TODO (BIGGEST PROBLEM WITH GLOBAL IMPURITY TREE): Appears that issue is split functions
    becoming "TOO" confident in their split, splatting lower node gradients to zero
    by nature.

    Experiment with the following as a split function:
        f(x) = 1{np.dot(theta, x) > 0}

        somehow...
'''


dec_tree = tree.DecisionTreeClassifier(max_depth = 4)
dec_tree = dec_tree.fit(X_train, y_train)
dec_tree_predictions = dec_tree.predict(X_test)
dec_tree_right = np.sum(dec_tree_predictions == y_test)

if X.shape[1] == 2:

    ax = plt.gca()
    bound_plotter.plot_contours(X, dec_tree.predict, ax, .0025)
    colors = [["blue", "red", "green"][y[i]] for i in range(y.shape[0])]
    plt.scatter(X[:,0], X[:,1], color = colors)
    plt.show()


model = GlobalImpurityModelTree2(head)
NUM_PROGRESS_ITERS = 25
GRID_STEP = 0.005
try:
    for display_progress_iter in range(NUM_PROGRESS_ITERS):
        try:

            model.train(X_train, y_train, 20000, 5, print_progress_iters = 100)
        except KeyboardInterrupt:
            print("display progress iter halted. Iters remaining: ", NUM_PROGRESS_ITERS - display_progress_iter)
        if X.shape[1] == 2:
            ax = plt.gca()
            bound_plotter.plot_contours(X, model.predict_leaves, ax, GRID_STEP)
            colors = [["blue", "red", "green"][y[i]] for i in range(y.shape[0])]
            plt.scatter(X[:,0], X[:,1], color = colors)
            plt.title("leaf predictions")
            plt.show()

            ax = plt.gca()
            bound_plotter.plot_contours(X, model.predict, ax, GRID_STEP)
            colors = [["blue", "red", "green"][y[i]] for i in range(y.shape[0])]
            plt.scatter(X[:,0], X[:,1], color = colors)
            plt.title("class predictions")
            plt.show()
except KeyboardInterrupt:
    print("training halted")



'''
#greedy model
model = greedy_impurity_tree_builder.build_logistic_impurity_tree(X.shape[1], 2)
model.train(None, X_train, y_train, 5, 1000000, [.01])
'''

model_predictions = model.predict(X_test)
model_right = np.sum(model_predictions==y_test)





print("model accuracy: ", 100.0*model_right/float(y_test.shape[0]))
print("decision tree accuracy: ", 100.0*dec_tree_right/float(y_test.shape[0]))




if X.shape[1] == 2:
    ax = plt.gca()
    bound_plotter.plot_contours(X, model.predict, ax, .0025)
    colors = [["blue", "red", "green"][y[i]] for i in range(y.shape[0])]
    plt.scatter(X[:,0], X[:,1], color = colors)
    plt.show()

    ax = plt.gca()
    bound_plotter.plot_contours(X, dec_tree.predict, ax, .0025)
    colors = [["blue", "red", "green"][y[i]] for i in range(y.shape[0])]
    plt.scatter(X[:,0], X[:,1], color = colors)
    plt.show()




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
