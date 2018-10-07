import numpy as np
import function.impurity as impurity
import toolbox.data_helper as data_helper
import optimize.general_gradient_descent as general_gradient_descent
from function.activation.sigmoid import Sigmoid

class LogisticImpurity:
    #TODO: Make this modular to use any child class of this that can provide a model gradient function of the required form

    #have this take a model_func once prototyping tested with hardcoded model
    def __init__(self):
        self.__theta = None

    def _set_theta(self, theta):
        self.__theta = theta

    def approx_gradient(self, X, y):
        def test_gini(params):
            model = LogisticImpurity()
            model._set_theta(params)
            return model.expected_gini(X, y)
        return general_gradient_descent.gradient(test_gini, self.__theta, 0.001)

    def expected_gini(self, X, y):
        probs = self.predict(X)
        subset_probs = np.array([probs, 1-probs]).T
        return impurity.expected_gini(subset_probs, y)

    def predict(self, X):
        assert(X.shape[1] == self.__theta.shape[0] or X.shape[1] == self.__theta.shape[0]+1)
        if X.shape[1] != self.__theta.shape[0]:
            X = data_helper.affine_X(X)
        return self.s0(np.dot(X, self.__theta))


    def train(self, X, y, steps, step_size):
        X = data_helper.affine_X(X)
        self.__rand_init_theta(X.shape[1])
        unique_labels = np.unique(y)
        for iter in range(steps):
            grad = self.gradient(X,y,unique_labels)
            self.__theta -= step_size * grad
            print("expected gini: ", self.expected_gini(X, y))
            print("grad: ", grad)
            print("------------------------------------------")

    def __rand_init_theta(self, features):
        self.__theta = np.random.rand(features)*.001

    def gradient(self, X, y, unique_labels):
        out = np.zeros(self.__theta.shape)
        for k in range(0, 2):
            s_k = self.s(np.dot(X, self.__theta), k)
            ds_k = self.ds_dx(s_k, k)
            u = self.u(s_k)
            v = self.v(s_k, y, unique_labels)
            du = self.du_dtheta(s_k, ds_k, k, X)
            dv = self.dv_dtheta(s_k, ds_k, k, X, y, unique_labels)
            out += du*v + dv*u
        return -out/float(X.shape[0])

    def u(self, s_outs):
        return 1.0/np.sum(s_outs)

    def v(self, s_outs, y, unique_labels):
        out = 0
        for l in unique_labels:
            out += (np.sum(s_outs[np.where(y==l)]))**2
        return out

    def du_dtheta(self, s_outs, ds_outs, k, X):
        left_divisor = np.sum(s_outs)
        left = -1.0/(left_divisor*left_divisor)
        return left * np.sum(X*ds_outs[:,np.newaxis], axis = 0)

    def dv_dtheta(self, s_outs, ds_outs, k, X, y, unique_labels):
        out = np.zeros(self.__theta.shape)
        for l in unique_labels:
            where_y_eq_l = np.where(y==l)
            left = np.sum(s_outs[where_y_eq_l])
            X_eq_l = X[where_y_eq_l]
            ds_eq_l = ds_outs[where_y_eq_l]
            right = np.sum(X_eq_l*ds_eq_l[:,np.newaxis], axis = 0)
            out += left * right
        return 2*out

    def s(self, X, k):
        return self.s0(X) if k == 0 else self.s1(X)

    def ds_dx(self, s_out, k):
        return self.ds0_dx(s_out) if k == 0 else self.ds1_dx(s_out)

    def s0(self, X):
        return 1.0/(1.0+np.exp(-X))

    def s1(self, X):
        return 1-self.s0(X)

    def ds0_dx(self, s0_out):
        return s0_out*(1-s0_out)

    def ds1_dx(self, s1_out):
        return -self.ds0_dx(s1_out)
