import model.impurity.global_impurity3.node_model3_maker as node_model3_maker
from model.impurity.global_impurity3.global_impurity_node3 import GlobalImpurityNode3
import sklearn.datasets as datasets
import toolbox.data_helper as data_helper
import timeit
import numpy as np
from model.impurity.global_impurity4.global_impurity_tree4 import GlobalImpurityTree4
import sys
import function.stable_func as stable_func

D_TYPE = np.float32

def num_nodes_in_depth(depth):
    out = 0
    for i in range(depth):
        out += 2**i
    return out

def init_params_tree(x_length, depth):
    return (0.0001*(np.random.rand(num_nodes_in_depth(depth), x_length) - .5)).astype(D_TYPE)

def d_sigmoid(X, sigmoid_outs):
    return sigmoid_outs*(1-sigmoid_outs)

np.random.seed(seed = 42)
X,y = datasets.load_digits(return_X_y = True)#datasets.load_iris(return_X_y = True)#
FEATURES = range(X.shape[1])#[0,1]#
X = X[:, FEATURES]
X = X.astype(D_TYPE)
#X/=16.0
NUM_POINTS = 4#X.shape[0]

X = X[0:NUM_POINTS,:]
y = y[:NUM_POINTS]
X = data_helper.affine_X(X)
unique_labels = np.unique(y)




tree = GlobalImpurityTree4(init_params_tree(X.shape[1], 3), stable_func.sigmoid, d_sigmoid)

start_time = timeit.default_timer()
splits = tree.calc_split_tree(X)
p_leaves = tree.calc_p_leaves(splits)
grad_splits = tree.calc_grad_split_tree(X, splits)
#grad_p_leaves = tree.calc_grad_p_leaves(p_leaves, splits, grad_splits)
grad_p_leaves_label_sums = tree.calc_label_sum_grad_p_leaves(p_leaves, splits, grad_splits, y, unique_labels)
grad_p_leaves_sums = tree.calc_sum_grad_p_leaves(grad_p_leaves_label_sums)
print("p_leaves calc'd in: ", timeit.default_timer() - start_time)
#print("grad_p_leaves: ", grad_p_leaves)
#print("grad_splits: ", grad_splits)
