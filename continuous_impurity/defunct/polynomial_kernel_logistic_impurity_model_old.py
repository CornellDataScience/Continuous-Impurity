from model.impurity.impurity_model2 import ImpurityModel2
import numpy as np
import toolbox.data_helper as data_helper
import function.stable_func as stable_func
'''
IS DEFUNCT BECAUSE I DIDN'T CONVERT TO THE DUAL FORM, NAIVELY JUST SET THE DOT PRODUCT TO THE KERNEL PRODUCT,
TOOK GRADIENTS THROUGH THAT. DOES NOT WORK AT ALL
'''
class PolynomialKernelLogisticImpurityModel(ImpurityModel2):
    #TODO: remove bias since the affine parameter will just account for the bias implicitly anyway (i.e. if bias is 2 and bias in theta wants to be 8, it'll just be 6 instead)
    def __init__(self, X_shape, bias, degree):
        ImpurityModel2.__init__(self, 2, (X_shape[1]+1))
        self.__bias = bias
        self.__degree = degree


    def _kernel(self, v1, v2, c, d):
        return np.float_power(np.dot(v1, v2)+c, d)

    '''
    assumes A is a matrix,
    assumes v2 is a vector s.t. A dot v2 is legal.
    gives the gradient of _kernel(v1, v2, c, d) w.r.t. v2 (a vector)
    '''
    def _d_kernel_d_v2(self, A, v2, c, d):
        return d*A*self._kernel(A, v2, c, d-1)[:,np.newaxis]

    '''
    returns a matrix, A, with shape (X.shape[0], self.__num_subgroups)
    where A[i,j] is the probability of X[i] being assigned to subset j.'''
    def predict(self, X):
        X_affine = data_helper.affine_X(X)
        p0 = stable_func.sigmoid(self._kernel(X_affine, self._theta, self.__bias, self.__degree))
        return np.column_stack([p0, 1-p0])


    '''
    returns a matrix A with shape (X.shape[0], self.__num_subgroups) + theta.shape
    s.t. A[i,k] = grad w.r.t. theta of p(k|X[i])
    passes all arguments, even if the implementation does not require them.
    '''
    def _d_predict_d_theta(self, X, predicts):
        X_affine = data_helper.affine_X(X)
        d_kernel_d_theta = self._d_kernel_d_v2(X_affine, self._theta, self.__bias, self.__degree)
        dp0 = d_kernel_d_theta*(predicts[:,0]*(1.0-predicts[:,0]))[:,np.newaxis]
        out = np.zeros((X.shape[0], 2) + (self._theta.shape), dtype = np.float64)
        out[:,0,:] = dp0
        out[:,1,:] = -dp0
        return out
