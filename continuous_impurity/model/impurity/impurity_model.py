import numpy as np
import function.impurity as impurity
import toolbox.data_helper as data_helper
import optimize.general_gradient_descent as general_gradient_descent
from function.activation.sigmoid import Sigmoid

class ImpurityModel:
    def __init__(self, model_func):
        self.__model_func = model_func

    def expected_gini(self, X, y):
        probs = self.predict(X)
        return impurity.expected_gini(probs, y)

    def predict(self, X):
        return self.__model_func.func(X)

    def train(self, X, y, steps, step_size):
        self.__model_func.rand_init_params(X)
        unique_labels = np.unique(y)
        for iter in range(steps):
            self.__model_func.step_params(-step_size*self.__gradient(X,y,unique_labels))
            if iter%10 == 0:
                print("expected gini: ", self.expected_gini(X, y))
                print("------------------------------------------")




    def __gradient(self, X, y, unique_labels):
        grad = np.zeros(self.__model_func.params_shape())
        model_outs = self.__model_func.func(X)
        for p in range(grad.shape[0]):
            for j in range(grad.shape[1]):
                for k in range(model_outs.shape[1]):
                    u = self.__u(model_outs, k)
                    v = self.__v(model_outs, y, unique_labels, k)
                    du = self.__du(model_outs, X, k, p, j)
                    dv = self.__dv(model_outs, X, y, unique_labels, k, p, j)
                    grad[p,j] += v*du + u*dv

        grad *= -1.0/float(X.shape[0])
        return grad

    def __u(self, model_outs, k):
        return 1.0/(np.sum(model_outs[:,k]))

    def __v(self, model_outs, y, unique_labels, k):
        out = 0
        for l in unique_labels:
            out +=  np.sum(model_outs[np.where(y==l),k])**2
        return out

    def __du(self, model_outs, X, k, p, j):
        left = -1.0/(np.sum(model_outs[:,k])**2)

        right = 0
        for i in range(model_outs.shape[0]):
            right += self.__model_func.d_func(X[i],k,p,j)
        return left*right

    def __dv(self, model_outs, X, y, unique_labels, k, p, j):
        out = 0
        for l in unique_labels:
            left = np.sum(model_outs[np.where(y==l), k])
            right = 0
            for i in np.where(y==l)[0]:
                right += self.__model_func.d_func(X[i],k,p,j)
            out += left*right
        return 2*out
