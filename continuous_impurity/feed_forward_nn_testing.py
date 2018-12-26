import model.impurity.global_impurity3.node_model3_maker as node_model3_maker
from model.impurity.global_impurity3.global_impurity_node3 import GlobalImpurityNode3
import sklearn.datasets as datasets
import toolbox.data_helper as data_helper
import timeit
import numpy as np
from model.impurity.global_impurity4.global_impurity_tree4 import GlobalImpurityTree4
import sys
import function.stable_func as stable_func
from function.activation.sigmoid import Sigmoid
from function.activation.relu import Relu
from function.activation.tanh import TanH
from model.nn.feed_forward_nn import FeedForwardNN
from model.nn.cost.square_error import SquareError

D_TYPE = np.float32


np.random.seed(seed = 42)
X,y = datasets.load_iris(return_X_y = True)#datasets.load_digits(return_X_y = True)#
X = X.astype(np.float32)
y_unq = np.unique(y)
y_new = np.zeros((y.shape[0], len(y_unq)), dtype = X.dtype)
for l_ind in range(len(y_unq)):
    y_new[np.where(y == y_unq[l_ind]), l_ind] = 1
y = y_new

FEATURES = range(X.shape[1])#
X = X[:, FEATURES]
#X = data_helper.unit_square_normalize(X)
X = X.astype(D_TYPE)
#X/=16.0
NUM_POINTS = X.shape[0]
(X,y),(a,b) = data_helper.train_test_split(X, y, .8)



layer_lengths = [X.shape[1],5,5,len(y_unq)]
act_funcs = [TanH() for i in range(len(layer_lengths)-2)]
act_funcs.append(Sigmoid())
cost = SquareError()
model = FeedForwardNN(layer_lengths, act_funcs, cost)
'''
forwards = model.forward(X[0])
#print("forwards: ", forwards)
print("forwards shapes: ", [arr.shape for arr in forwards])
backwards = model.backward(forwards)
print("backwards: ", backwards)
cost_grad = model.cost_grad(X[0], y[0])
print("cost_grad: ", cost_grad)

'''

LEARN_RATE = 0.1
BIAS_LEARN_RATE = LEARN_RATE * 2

for iters in range(0, 100000000):
    iter_cost = 0
    for i in range(len(X)):
        grad_A, grad_b = model.cost_grad(X[i], y[i])
        #print("grad_b: ", grad_b)
        #print("grad_A: ", grad_A)
        for l in range(len(grad_A)):
            model._A[l] += LEARN_RATE*grad_A[l]
            model._b[l] += BIAS_LEARN_RATE*grad_b[l]
        forwards = model.forward(X[i])
        y_hat = forwards[-1]
        #print("y_hat: ", y_hat)
        iter_cost += cost.cost(y[i], y_hat)
    print("ITER COST: ", iter_cost/float(X.shape[0]))
