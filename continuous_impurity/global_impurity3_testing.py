import model.impurity.global_impurity3.node_model3_maker as node_model3_maker
from model.impurity.global_impurity3.global_impurity_node3 import GlobalImpurityNode3
import sklearn.datasets as datasets
import toolbox.data_helper as data_helper
import timeit
import numpy as np
from model.impurity.global_impurity3.global_impurity_model_tree3 import GlobalImpurityModelTree3

np.random.seed(seed = 42)
X,y = datasets.load_digits(return_X_y = True)#datasets.load_iris(return_X_y = True)#
FEATURES = range(X.shape[1])#[0,1]
X = X[:, FEATURES]
X = X.astype(np.float64)
X/=16.0
NUM_POINTS = X.shape[0]

X = X[0:NUM_POINTS,:]
y = y[:NUM_POINTS]
X = data_helper.affine_X(X)
unique_labels = np.unique(y)
tree = GlobalImpurityModelTree3(node_model3_maker.logistic_model_at_depth(X.shape[1]))


tree.train(X, y, 10.0, 100000, 1, 5, 0, 0, 5, iters_per_prune = 5, print_progress_iters = 5)
