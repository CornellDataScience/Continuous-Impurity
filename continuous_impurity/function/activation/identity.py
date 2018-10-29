from function.activation.activation_function import ActivationFunction
import function.stable_func as stable_func
import numpy as np

'''
The identity function as an activation function. should ONLY be used for testing (since activation functions are generally used to be nonlinear)
'''
class Identity(ActivationFunction):

    def act(self, X):
        return X

    def derivative_wrt_activation(self, act_outs):
        return np.zeros(act_outs.shape, dtype = np.float64)
