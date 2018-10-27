from model.impurity.impurity_model2 import ImpurityModel2
import numpy as np
import toolbox.data_helper as data_helper
import function.stable_func as stable_func

'''TODO: Make this class also abstract allowing for overwriting the kernel function and its derivative/s.
Vanilla logistic impurity could just use the normal dot product for the kernel.
'''
class LogisticImpurityModel2(ImpurityModel2):

    def __init__(self, X_shape):
        ImpurityModel2.__init__(self, 2, (X_shape[1]+1))




    '''returns a matrix, A, with shape (X.shape[0], self.__num_subgroups)
    where A[i,j] is the probability of X[i] being assigned to subset j.'''
    def predict(self, X):
        X_affine = data_helper.affine_X(X)
        p0 = stable_func.sigmoid(np.dot(X_affine, self._theta))
        return np.column_stack([p0, 1-p0])

    '''
    returns a matrix A with shape (X.shape[0], self.__num_subgroups) + theta.shape
    s.t. A[i,k] = grad w.r.t. theta of p(k|X[i])
    passes all arguments, even if the implementation does not require them.
    '''
    def _d_predict_d_theta(self, X, predicts):
        X_affine = data_helper.affine_X(X)
        dp0 = X_affine*(predicts[:,0]*(1.0-predicts[:,0]))[:,np.newaxis]
        out = np.zeros((X_affine.shape[0], 2) + self._theta.shape, dtype = np.float32)
        out[:,0,:] = dp0
        out[:,1,:] = -dp0
        return out
