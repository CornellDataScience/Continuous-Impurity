from abc import ABC, abstractmethod
import numpy as np
import function.impurity as impurity
from time import sleep
class ImpurityModel2(ABC):

    def __init__(self, num_subgroups, theta_shape):
        self._theta = np.random.random(theta_shape)-.5
        self._theta = self._theta.astype(np.float64)*.00001
        self.__num_subgroups = num_subgroups
    '''returns a matrix, A, with shape (X.shape[0], self.__num_subgroups)
    where A[i,j] is the probability of X[i] being assigned to subset j.'''
    @abstractmethod
    def predict(self, X):
        pass

    '''
    returns a matrix A with shape (X.shape[0], self.__num_subgroups) + theta.shape
    s.t. A[i,k] = grad w.r.t. theta of p(k|X[i])
    passes all arguments, even if the implementation does not require them.
    '''
    @abstractmethod
    def _d_predict_d_theta(self, X, predicts):
        pass

    def train(self, X, y, iters, step_size, print_progress_iters = 1000):
        unique_labels = np.unique(y)
        for iter in range(iters):
            grad = self.__gradient(X, y, unique_labels)
            self._theta -= step_size*grad
            if iter%print_progress_iters == 0:
                print("iter: ", iter)
                print("expected gini: ", self.__expected_gini(X,y))
                print("grad step: ", grad_step)
                print("theta: ", self._theta)
                print("theta - grad step: ", self._theta - grad_step)
                print("---------------------------------------------")


    def __expected_gini(self, X, y):
        return impurity.expected_gini(self.predict(X), y)

    def __gradient(self, X, y, unique_labels):
        out = np.zeros(self._theta.shape, dtype = np.float64)
        predicts = self.predict(X)
        d_predicts = self._d_predict_d_theta(X, predicts)
        u = self.__u(predicts)
        du = self.__du_dtheta(predicts, d_predicts)
        v = self.__v(predicts, y, unique_labels)
        dv = self.__dv_dtheta(predicts, d_predicts, y, unique_labels)

        for k in range(self.__num_subgroups):
            u_k = u[k]
            du_k = du[k]
            v_k = v[k]
            dv_k = dv[k]
            out += v_k*du_k + u_k*dv_k
        return -out/float(X.shape[0])

    '''
    expects d_predicts to have shape (predicts.shape) + (theta.shape),
    so that d_predicts[i,k] = grad w.r.t. theta of p(k|X[i]). (i.e. they should be from
    _d_predict_d_theta)

    outputs a matrix dU with shape (predicts.shape[1],) + (d_predicts.shape[2], d_predicts.shape[3])
    s.t. dU[k] is grad w.r.t. theta of u_k
    '''
    def __du_dtheta(self, predicts, d_predicts):
        out = np.full(d_predicts.shape[1:], -1.0, dtype = np.float64)
        out *= np.sum(d_predicts, axis = 0)
        pred_sums = np.sum(predicts, axis = 0)
        #done for numerical stability (divide by 0)
        out[np.where(pred_sums == 0)] = 0
        out[np.where(pred_sums != 0)] /= np.square(pred_sums[np.where(pred_sums != 0)])[:,np.newaxis]
        return out

    '''
    expects d_predicts to have shape (predicts.shape) + (theta.shape),
    so that d_predicts[i,k] = grad w.r.t. theta of p(k|X[i]). (i.e. they should be from
    _d_predict_d_theta)

    outputs a matrix dV with shape (predicts.shape[1],) + (d_predicts.shape[2], d_predicts.shape[3])
    s.t. dV[k] is grad w.r.t. theta of u_k
    '''
    def __dv_dtheta(self, predicts, d_predicts, y, unique_labels):
        out = np.zeros(d_predicts.shape[1:], dtype = np.float64)
        for l in unique_labels:
            where_y_eq_l = np.where(y==l)
            out += np.sum(d_predicts[where_y_eq_l], axis = 0)*(np.sum(predicts[where_y_eq_l], axis = 0))[:,np.newaxis]
        return 2*out

    def __u(self, predicts):
        sums = np.sum(predicts, axis = 0)
        #done for stability (divide by 0)
        out = np.zeros(sums.shape, dtype = np.float64)
        out[np.where(sums != 0)] = 1.0/sums[np.where(sums != 0)]
        return out

    def __v(self, predicts, y, unique_labels):
        out = np.zeros(predicts.shape[1], np.float64)
        for l in unique_labels:
            where_y_eq_l = np.where(y == l)
            out += np.square(np.sum(predicts[where_y_eq_l], axis = 0))
        return out
