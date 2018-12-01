from function.activation.activation_function import ActivationFunction
import function.stable_func as stable_func
import numpy as np
class Relu(ActivationFunction):

    def act(self, X):
        out = X.copy()
        out[np.where(X < 0)] = 0
        return out

    def derivative_wrt_activation(self, act_outs):
        out = np.zeros(act_outs.shape)
        out[np.where(act_outs != 0)] = 1
        return out
