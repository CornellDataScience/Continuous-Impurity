import model.impurity.global_impurity3.node_model3_maker as node_model3_maker
from model.impurity.global_impurity3.global_impurity_node3 import GlobalImpurityNode3
import sklearn.datasets as datasets
import toolbox.data_helper as data_helper
import timeit
import numpy as np

X,y = datasets.load_iris(return_X_y = True)#datasets.load_digits(return_X_y = True)#
FEATURES = range(X.shape[1])#[0,1]
X = X[:, FEATURES]

NUM_POINTS = X.shape[0]

X = X[0:NUM_POINTS,:]
y = y[:NUM_POINTS]
X = data_helper.affine_X(X)
unique_labels = np.unique(y)

def make_tree(model_maker, max_depth):
    def f(node, d):
        if d >= max_depth:
            return None
        children = [GlobalImpurityNode3(node, model_maker(d) if d < max_depth - 1 else None, children = []) for i in range(2)]
        node._add_children(children)
        for child in children:
            f(child, d+1)

    head = GlobalImpurityNode3(None, model_maker(0), children = [])
    f(head, 1)
    return head

tree = make_tree(node_model3_maker.logistic_model_at_depth(X.shape[1]), 4)
start_time = timeit.default_timer()
#print("tree built")
tree_list = GlobalImpurityNode3.to_list_and_set_IDs(tree)
leaves = GlobalImpurityNode3.get_leaves(tree_list)
f_arr = GlobalImpurityNode3.calc_f_arr(tree_list, X)
#print("f_arr: ", f_arr)
grad_f_arr = GlobalImpurityNode3.calc_grad_f_arr(tree_list, X, f_arr)
#print("grad_f_arr: ", grad_f_arr)
p_arr = GlobalImpurityNode3.calc_p_arr(tree_list, X, f_arr)
#print("p_arr: ", p_arr)


for leaf in leaves:
    grad_p_leaf_arr = GlobalImpurityNode3.calc_grad_p_arr(tree_list, leaf._ID, p_arr, f_arr, grad_f_arr)
    #print("grad_p_leaf_arr: ", grad_p_leaf_arr)
print("grad run time: ", timeit.default_timer() - start_time)


grad_EG = GlobalImpurityNode3.calc_grad(X, y, unique_labels, tree_list, leaves, f_arr, p_arr, grad_f_arr)

tree.train(X, y, .1, 100000)
