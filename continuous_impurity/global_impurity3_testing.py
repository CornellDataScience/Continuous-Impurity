import model.impurity.global_impurity3.node_model3_maker as node_model3_maker
from model.impurity.global_impurity3.global_impurity_node3 import GlobalImpurityNode3
import sklearn.datasets as datasets
import toolbox.data_helper as data_helper
import timeit
import numpy as np

X,y = datasets.load_digits(return_X_y = True)#datasets.load_iris(return_X_y = True)#
FEATURES = range(X.shape[1])#[0,1]
X = X[:, FEATURES]

NUM_POINTS = X.shape[0]//4

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

tree = make_tree(node_model3_maker.logistic_model_at_depth(X.shape[1]), 10)



tree.train(X, y, .1, 100000, GC_frequency = None)
