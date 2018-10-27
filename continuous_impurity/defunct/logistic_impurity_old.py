import numpy as np
import function.impurity as impurity
import toolbox.data_helper as data_helper
import optimize.general_gradient_descent as general_gradient_descent
import function.stable_func as stable_func


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

    def gini(self, X, y):
        probs = self.predict(X)
        lefts = y[np.where(probs<=.5)]
        rights = y[np.where(probs>.5)]
        print("lefts: ", lefts)
        print("rights: ", rights)
        return impurity.gini([lefts, rights])


    def predict(self, X):
        #clean the single input vs. multi input vs. not-affined input handling up a lot
        if len(X.shape) == 1:
            X = np.array([X])
        assert(X.shape[1] == self.__theta.shape[0] or X.shape[1] == self.__theta.shape[0]-1)
        if X.shape[1] != self.__theta.shape[0]:
            X = data_helper.affine_X(X)
        out = self.__s(np.dot(X, self.__theta), 0)
        return out[0] if X.shape[0] == 1 else out


    def train(self, X, y, steps, step_size):
        X = data_helper.affine_X(X)
        self.__rand_init_theta(X.shape[1])
        unique_labels = np.unique(y)
        for iter in range(steps):
            grad = self.__gradient(X,y,unique_labels)
            self.__theta -= step_size * grad
            if iter%1000 == 0:
                print("iter: ", iter)
                print("expected gini: ", self.expected_gini(X, y))
                print("actual gini: ", self.gini(X, y))
                print("------------------------------------------")

    def __rand_init_theta(self, features):
        self.__theta = np.random.rand(features)*.001

    def __gradient(self, X, y, unique_labels):
        out = np.zeros(self.__theta.shape)
        for k in range(0, 2):
            s_k = self.__s(np.dot(X, self.__theta), k)
            ds_k = self.__ds_dx(s_k, k)
            out += self.__du_dtheta(s_k,ds_k,k,X)*self.__v(s_k,y,unique_labels) \
                + self.__dv_dtheta(s_k, ds_k,k,X,y,unique_labels)*self.__u(s_k)
        return -out/float(X.shape[0])

    def __u(self, s_outs):
        return 1.0/np.sum(s_outs)

    def __v(self, s_outs, y, unique_labels):
        out = 0
        for l in unique_labels:
            out += (np.sum(s_outs[np.where(y==l)]))**2
        return out

    def __du_dtheta(self, s_outs, ds_outs, k, X):
        return -np.sum(X*ds_outs[:,np.newaxis], axis = 0)/(np.square(np.sum(s_outs)))

    def __dv_dtheta(self, s_outs, ds_outs, k, X, y, unique_labels):
        out = np.zeros(self.__theta.shape)
        for l in unique_labels:
            where_y_eq_l = np.where(y==l)
            out += np.sum(s_outs[where_y_eq_l])*np.sum(X[where_y_eq_l]*(ds_outs[where_y_eq_l][:,np.newaxis]), axis = 0)
        return 2*out

    def __s(self, X, k):
        s0 = stable_func.sigmoid(X)
        return s0 if k == 0 else 1-s0

    def __ds_dx(self, s_out, k):
        ds0 = s_out*(1-s_out)
        return ds0 if k == 0 else -ds0
