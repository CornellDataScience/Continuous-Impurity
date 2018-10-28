from model.impurity.impurity_model import ImpurityModel
import numpy as np
import toolbox.data_helper as data_helper
import function.stable_func as stable_func

class LogisticImpurityModel(ImpurityModel):

    def __init__(self, x_length):
        ImpurityModel.__init__(self, 2, [(x_length+1)])




    '''returns a matrix, A, with shape (X.shape[0], self.__num_subgroups)
    where A[i,j] is the probability of X[i] being assigned to subset j.'''
    def predict(self, X):
        X_affine = data_helper.affine_X(X)
        p0 = stable_func.sigmoid(np.dot(X_affine, self._get_params()[0]))
        return np.column_stack([p0, 1-p0])


    def _d_predict_d_params(self, X, predicts):
        X_affine = data_helper.affine_X(X)
        dp0 = X_affine*(predicts[:,0]*(1.0-predicts[:,0]))[:,np.newaxis]
        out = np.zeros((X_affine.shape[0], 2) + self._get_params()[0].shape, dtype = np.float64)
        out[:,0,:] = dp0
        out[:,1,:] = -dp0
        return [out]
