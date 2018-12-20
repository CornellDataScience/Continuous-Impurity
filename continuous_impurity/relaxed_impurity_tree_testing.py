import model.impurity.global_impurity3.node_model3_maker as node_model3_maker
from model.impurity.global_impurity3.global_impurity_node3 import GlobalImpurityNode3
import sklearn.datasets as datasets
import toolbox.data_helper as data_helper
import timeit
import numpy as np
from model.impurity.global_impurity3.global_impurity_model_tree3 import GlobalImpurityModelTree3
from model.impurity.relaxed_impurity.relaxed_global_sigmoid_impurity_tree import RelaxedGlobalSigmoidImpurityTree
import sys

np.random.seed(seed = 42)
X,y = datasets.load_iris(return_X_y = True)#
FEATURES = range(X.shape[1])#[0,1]
X = X[:, FEATURES]
X = X.astype(np.float64)
NUM_POINTS = X.shape[0]

X = X[0:NUM_POINTS,:]
y = y[:NUM_POINTS]
X = data_helper.affine_X(X)
tree = RelaxedGlobalSigmoidImpurityTree(X.shape[1], 2)#GlobalImpurityModelTree3(node_model3_maker.logistic_model_at_depth(X.shape[1]))
tree.calc_leaf_probs(X[0], 1)
