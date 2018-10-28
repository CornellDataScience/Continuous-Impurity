from model.impurity.impurity_model import ImpurityModel
import numpy as np
import toolbox.data_helper as data_helper
import function.stable_func as stable_func
from model.parameterized_transform.matrix_activation_transform import MatrixActivationTransform
import timeit

class MatrixActivationLogisticImpurity(ImpurityModel):

    def __init__(self, act_func, x_length, transform_x_length):
        self.__act_func = act_func
        mat_shape = (transform_x_length, x_length+1)
        ImpurityModel.__init__(self, 2, [(transform_x_length), mat_shape])

    def __get_mat_act_transform(self):
        return MatrixActivationTransform(self.__act_func, self._get_params()[1])

    '''returns a matrix, A, with shape (X.shape[0], self.__num_subgroups)
    where A[i,j] is the probability of X[i] being assigned to subset j.'''
    def predict(self, X):
        X_affine = data_helper.affine_X(X)
        transforms = self.__get_mat_act_transform().transform(X_affine)
        p0 = stable_func.sigmoid(np.dot(transforms, self._get_params()[0]))
        return np.column_stack([p0, 1-p0])


    def _d_predict_d_params(self, X, predicts):
        times = []

        X_affine = data_helper.affine_X(X)

        start_time = timeit.default_timer()
        transform_X_affine = self.__get_mat_act_transform().transform(X_affine)
        times.append(timeit.default_timer()-start_time)


        start_time = timeit.default_timer()
        sigmoid_deriv = predicts[:,0]*(1.0-predicts[:,0])
        times.append(timeit.default_timer()-start_time)


        start_time = timeit.default_timer()
        grad_predict0_params0 = transform_X_affine*(sigmoid_deriv[:,np.newaxis])
        times.append(timeit.default_timer()-start_time)


        start_time = timeit.default_timer()
        mat_act_derivs = self.__get_mat_act_transform().param_grad(X_affine, transform_X_affine)
        times.append(timeit.default_timer()-start_time)



        grad_predict0_params1 = np.zeros((X.shape[0],) + self._get_params()[1].shape, dtype = np.float64)

        start_time = timeit.default_timer()
        #slowest

        for j in range(grad_predict0_params1.shape[1]):
            grad_predict0_params1[:,j] = sigmoid_deriv[:,np.newaxis]*\
                self._get_params()[0][j]*\
                mat_act_derivs[:,j,j]


        start_time = timeit.default_timer()
        grad_predict0_params = [grad_predict0_params0, grad_predict0_params1]
        out = []
        for i in range(2):
            outi = np.zeros((X_affine.shape[0], 2) + self._get_params()[i].shape, dtype = np.float64)
            outi[:,0] = grad_predict0_params[i]
            outi[:,1] = -grad_predict0_params[i]
            out.append(outi)
        times.append(timeit.default_timer()-start_time)

        times = np.asarray(times)
        #print("times: ", times)
        #print("relative times: ", times/np.sum(times))
        #print("--------------------------------------------------------------------------")
        return out
