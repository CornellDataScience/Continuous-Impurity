from model.parameterized_transform.parameterized_transform import ParameterizedTransform
from function.activation.activation_function import ActivationFunction
import numpy as np

class MatrixActivationTransform(ParameterizedTransform):

    '''
    where mat_shape creates a matrix with that shape, A, s.t.
    Ax (x being a single, column vector) is legal
    '''
    def __init__(self, act_func, params):
        assert isinstance(act_func, ActivationFunction)
        self.__act_func = act_func
        ParameterizedTransform.__init__(self, params)

    '''
    Transforms X, where X is a matrix whose rows are the vectors to be transformed.
    '''
    def transform(self, X):
        return self.__act_func.act(np.dot(self.params, X.T)).T




    '''
    Returns a matrix, A, with shape transform(X).shape + params.shape
    s.t. A[i,j,k,l] is partial (T(X[i]))[j]/ partial params[k,l]
    (in this case A[_,j,k,_] is non-zero only if j = k, so will
    be fairly sparse)
    '''
    def param_grad(self, X, transform_outs):
        grad = np.zeros((X.shape[0], self.params.shape[0]) + self.params.shape, dtype = np.float64)
        act_derivs = self.__act_func.derivative_wrt_activation(transform_outs)
        for i in range(grad.shape[0]):
            for j in range(grad.shape[1]):
                for l in range(grad.shape[3]):
                    grad[i,j,j,l] = X[i,l]*self.__act_func.derivative_wrt_activation(np.dot(self.params[j], X[i]))
        return grad
