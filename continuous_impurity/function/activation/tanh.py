from function.activation.activation_function import ActivationFunction
import function.stable_func as stable_func
import numpy as np
class TanH(ActivationFunction):

    def act(self, X):
        return stable_func.tanh(X)

    def derivative_wrt_activation(self, act_outs):
        return 1-(act_outs * act_outs)
