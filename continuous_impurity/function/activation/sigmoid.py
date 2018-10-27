from function.activation.activation_function import ActivationFunction
import function.stable_func as stable_func
class Sigmoid(ActivationFunction):

    def act(self, X):
        return stable_func.sigmoid(X)
    
    def derivative_wrt_activation(self, act_outs):
        return act_outs*(1-act_outs)
