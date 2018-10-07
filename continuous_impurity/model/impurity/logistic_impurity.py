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

    def train(self, X, y, steps, step_size):
        X = data_helper.affine_X(X)
        self.__theta = np.random.rand((X.shape[1]))*.001
        unique_labels = np.unique(y)
        for iter in range(steps):
            grad = self.gradient(X,y,unique_labels)
            self.__theta -= step_size * grad
            print("expected gini: ", self.expected_gini(X, y))
            print("grad: ", grad)
            print("------------------------------------------")

    def gradient(self, X, y, unique_labels):
        out = np.zeros(self.__theta.shape)
        for k in range(0, 2):
            u = self.u(k, X)
            v = self.v(k, X, y, unique_labels)
            for j in range(out.shape[0]):
                du = self.du_dtheta(k, j, X)
                dv = self.dv_dtheta(k, j, X, y, unique_labels)
                out[j] += du*v + dv*u
        return -out/float(X.shape[0])

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
        out = np.zeros(X.shape[0])
        for i in range(out.shape[0]):
            out[i] = self.s0(np.dot(self.__theta,  X[i]))
        return out

    def u(self, k, X):
        sum = 0
        for i in range(X.shape[0]):
            sum += self.s(k, np.dot(self.__theta, X[i]))
        return 1.0/sum

    def du_dtheta(self, k, j, X):
        left = 0
        for i in range(X.shape[0]):
            left += self.s(k, np.dot(self.__theta, X[i]))
        left = -1.0/(left*left)

        right = 0
        for i in range(X.shape[0]):
            s_out = self.s(k, np.dot(self.__theta, X[i]))
            right += X[i,j]*self.ds_dx(k, s_out)

        return left * right

    def v(self, k, X, y, unique_labels):
        out = 0
        for l in unique_labels:
            iter_sum = 0
            for i in range(X.shape[0]):
                iter_sum += 0 if y[i] != l else self.s(k, np.dot(self.__theta, X[i]))
            out += iter_sum*iter_sum
        return out

    def dv_dtheta(self, k, j, X, y, unique_labels):
        out = 0
        for l in unique_labels:
            left = 0
            right = 0
            for i in range(0, X.shape[0]):
                if y[i] == l:
                    s_out = self.s(k, np.dot(self.__theta, X[i]))
                    left += s_out
                    right += X[i,j]*self.ds_dx(k, s_out)
            out += left * right
        out *= 2
        return out



    def s(self, k, X):
        return self.s0(X) if k == 0 else self.s1(X)

    def ds_dx(self, k, s_out):
        return self.ds0_dx(s_out) if k == 0 else self.ds1_dx(s_out)

    def s0(self, X):
        return 1.0/(1.0+np.exp(-X))

    def s1(self, X):
        return 1-self.s0(X)

    def ds0_dx(self, s0_out):
        return s0_out*(1-s0_out)

    def ds1_dx(self, s1_out):
        return -self.ds0_dx(s1_out)
