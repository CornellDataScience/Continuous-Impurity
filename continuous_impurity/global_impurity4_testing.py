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
    return (0.00001*(np.random.rand(num_nodes_in_depth(depth), x_length) - .5)).astype(D_TYPE)

def d_sigmoid(X, sigmoid_outs):
    return sigmoid_outs*(1-sigmoid_outs)

np.random.seed(seed = 42)
X,y = datasets.load_digits(return_X_y = True)#datasets.load_iris(return_X_y = True)#
FEATURES = range(X.shape[1])#[0,1]#
X = X[:, FEATURES]
X = X.astype(D_TYPE)
X/=16.0
NUM_POINTS = X.shape[0]

X = X[0:NUM_POINTS,:]
y = y[:NUM_POINTS]
X = data_helper.affine_X(X)
unique_labels = np.unique(y)
where_y_eq_ls = []
for l in unique_labels:
    where_y_eq_ls.append(np.where(y == l)[0])



tree = GlobalImpurityTree4(init_params_tree(X.shape[1], 5), stable_func.sigmoid, d_sigmoid)


'''
TODO: Make sure that transition from accidentally not treating leaves as leaves, but last model nodes as leaves,
    have correct for loop bounds
'''



start_time = timeit.default_timer()

tree.train(X,y,50000,50)

print("model trained in: ", timeit.default_timer() - start_time)
'''
grad_p_leaves_label_sums = tree.calc_label_sum_grad_p_leaves(p_leaves, splits, grad_splits, where_y_eq_ls)
#print("grad_p_leaves_label_sums: ", grad_p_leaves_label_sums.shape)
#print("GRAD_P_LEAVES_LABEL_SUMS: ", grad_p_leaves_label_sums)

#grad_gini = tree.calc_expected_gini_gradient(p_leaves, grad_p_leaves_label_sums, where_y_eq_ls)
#print("grad_gini: ", grad_gini.shape)
#print("grad_gini: ", grad_gini)



print("p_leaves calc'd in: ", timeit.default_timer() - start_time)
tree.train(X, y, 10000, .001)
#print("grad_p_leaves: ", grad_p_leaves)
#print("grad_splits: ", grad_splits)
'''
