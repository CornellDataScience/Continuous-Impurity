from abc import ABC, abstractmethod
import numpy as np
import function.impurity as impurity
from time import sleep
class ImpurityModel(ABC):

    def __init__(self, num_subgroups, params_shapes):
        self.__params = []
        for i in range(len(params_shapes)):
            self.__params.append(0.00001*(np.random.random(params_shapes[i])-.5).astype(np.float64))
        self.__num_subgroups = num_subgroups

    '''returns a matrix, A, with shape (X.shape[0], self.__num_subgroups)
    where A[i,j] is the probability of X[i] being assigned to subset j.'''
    @abstractmethod
    def predict(self, X):
        pass

    '''
    returns a list, L, of numpy arrays such that L[n] is the gradient of the model,
    A, w.r.t. the nth paramater (vector, matrix, etc.), where A is of the shape:
    (self.__num_subgroups, X.shape[0]) + self.__params[n].shape
    '''
    @abstractmethod
    def _d_predict_d_params(self, X, predicts):
        pass

    def _get_params(self):
        return self.__params

    def train(self, X, y, iters, step_sizes, print_progress_iters = 1000):
        unique_labels = np.unique(y)
        for iter in range(iters):
            grad = self.__gradient(X, y, unique_labels)
            for i in range(len(grad)):
                self.__params[i] -= step_sizes[i]*grad[i]
            if iter%print_progress_iters == 0:
                print("iter: ", iter)
                print("expected gini: ", self.__expected_gini(X,y))
                print("params: ", self.__params)
                print("grad: ", grad)
                print("---------------------------------------------")



    def __gradient(self, X, y, unique_labels):
        out = [np.zeros(param.shape, dtype = np.float64) for param in self.__params]
        predicts = self.predict(X)
        d_params = self._d_predict_d_params(X, predicts)
        u = self.__u(predicts)
        du = self.__du_dthetas(predicts, d_params)
        v = self.__v(predicts, y, unique_labels)
        dv = self.__dv_dthetas(predicts, d_params, y, unique_labels)
        for i in range(len(out)):
            for k in range(self.__num_subgroups):
                u_k = u[k]
                du_k = du[i][k]
                v_k = v[k]
                dv_k = dv[i][k]
                out[i] += v_k*du_k + u_k*dv_k
            out[i] /= -float(X.shape[0])

        return out


    def __expected_gini(self, X, y):
        return impurity.expected_gini(self.predict(X), y)

    '''
    expects d_predicts to have shape (predicts.shape) + (theta.shape),
    so that d_predicts[i,k] = grad w.r.t. theta of p(k|X[i]). (i.e. they should be from
    _d_predict_d_theta)

    outputs a matrix dU with shape (predicts.shape[1],) + (d_predicts.shape[2], d_predicts.shape[3])
    s.t. dU[k] is grad w.r.t. theta of u_k
    '''
    def __du_dthetas(self, predicts, d_predicts):
        out = []
        for i in range(len(d_predicts)):
            out.append(np.full(d_predicts[i].shape[1:], -1.0, dtype = np.float64))

        '''
        pred_sums = np.sum(predicts, axis = 0)
        print("pred_sums: ", pred_sums)
        print("predicts: ", predicts)

        for i in range(len(out)):
            out[i] *= np.sum(d_predicts[i], axis = 0)
            #done for numerical stability (divide by 0)
            out[i][np.where(pred_sums == 0)] = 0
            out[i][np.where(pred_sums != 0)] /= np.square(pred_sums[np.where(pred_sums != 0)])[:,np.newaxis]
        '''

        sqrd_pred_sums = np.square(np.sum(predicts, axis = 0))
        for i in range(len(out)):
            for j in range(sqrd_pred_sums.shape[0]):
                if sqrd_pred_sums[j] != 0:
                    out[i] *= np.sum(d_predicts[i], axis = 0)/sqrd_pred_sums[j]

        return out

    '''
    expects d_predicts to have shape (predicts.shape) + (theta.shape),
    so that d_predicts[i,k] = grad w.r.t. theta of p(k|X[i]). (i.e. they should be from
    _d_predict_d_theta)

    outputs a matrix dV with shape (predicts.shape[1],) + (d_predicts.shape[2], d_predicts.shape[3])
    s.t. dV[k] is grad w.r.t. theta of u_k
    '''
    def __dv_dthetas(self, predicts, d_predicts, y, unique_labels):
        out = []
        for i in range(len(d_predicts)):
            out.append(np.zeros(d_predicts[i].shape[1:], dtype = np.float64))

        for l in unique_labels:
            where_y_eq_l = np.where(y==l)
            for i in range(len(d_predicts)):
                d_predicts_lsum = np.sum(d_predicts[i][where_y_eq_l], axis = 0)
                predicts_lsum = np.sum(predicts[where_y_eq_l], axis = 0)
                #adds new axises to predicts_lsum so that predicts_lsum*d_predicts
                #doesn't break, and also allows the result, B, to be:
                #B[i] = predicts_lsum[i]*d_predicts_lsum[i]
                while len(predicts_lsum.shape) != len(d_predicts_lsum.shape):
                    predicts_lsum = predicts_lsum = np.expand_dims(predicts_lsum, axis = len(predicts_lsum.shape))
                out[i] += d_predicts_lsum * predicts_lsum

        for i in range(len(out)):
            out[i] *= 2

        return out

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
