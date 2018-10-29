from model.impurity.impurity_model import ImpurityModel
import numpy as np
import toolbox.data_helper as data_helper
import function.stable_func as stable_func
from model.parameterized_transform.matrix_activation_transform import MatrixActivationTransform
import timeit

class MatrixActivationLogisticImpurity(ImpurityModel):
    TRANSFORM_BIAS_GRADIENT_DAMPEN_FACTOR = 0.00001
    def __init__(self, act_func, x_length, transform_x_length):
        self.__act_func = act_func
        mat_shape = (transform_x_length, x_length+1)
        param0 = 0.00001*(np.random.random(x_length+1)-.5).astype(np.float64)
        param1 = 0.00001*(np.random.random(mat_shape)-.5).astype(np.float64)
        for i in range(x_length):
            param1[i,i] = 1
        ImpurityModel.__init__(self, 2, [param0, param1])

    def __transform_prepare_X(self, X):
        return data_helper.affine_X(X)

    def __dot_prepare_transform(self, transformed_X):
        return data_helper.affine_X(transformed_X)#transformed_X

    def _get_mat_act_transform(self):
        return MatrixActivationTransform(self.__act_func, self._get_params()[1])

    '''returns a matrix, A, with shape (X.shape[0], self.__num_subgroups)
    where A[i,j] is the probability of X[i] being assigned to subset j.'''
    def predict(self, X):
        X_prep = self.__transform_prepare_X(X)
        transforms = self._get_mat_act_transform().transform(X_prep)
        transforms_prep = self.__dot_prepare_transform(transforms)
        p0 = stable_func.sigmoid(np.dot(transforms_prep, self._get_params()[0]))
        return np.column_stack([p0, 1-p0])


    def _d_predict_d_params(self, X, predicts):
        X_prep = self.__transform_prepare_X(X)
        transform_X_prep = self.__dot_prepare_transform(self._get_mat_act_transform().transform(X_prep))
        sigmoid_deriv = predicts[:,0]*(1.0-predicts[:,0])
        grad_predict0_params0 = transform_X_prep*(sigmoid_deriv[:,np.newaxis])



        grad_predict0_params1 = np.zeros((X.shape[0],) + self._get_params()[1].shape, dtype = np.float64)
        mat_act_derivs = self._get_mat_act_transform().param_grad(X_prep, transform_X_prep)
        for j in range(grad_predict0_params1.shape[1]):
            grad_predict0_params1[:,j] = (sigmoid_deriv[:,np.newaxis])*self._get_params()[0][j]*mat_act_derivs[:,j,j]

        grad_predict0_params1[:,grad_predict0_params1.shape[1]-1] *= MatrixActivationLogisticImpurity.TRANSFORM_BIAS_GRADIENT_DAMPEN_FACTOR


        grad_predict0_params = [grad_predict0_params0, grad_predict0_params1]
        out = []
        for i in range(2):
            outi = np.zeros((X_prep.shape[0], 2) + self._get_params()[i].shape, dtype = np.float64)
            outi[:,0] = grad_predict0_params[i]
            outi[:,1] = -grad_predict0_params[i]
            out.append(outi)
        return out
