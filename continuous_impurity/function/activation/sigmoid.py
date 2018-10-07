from function.activation.activation_function import ActivationFunction

class Sigmoid(ActivationFunction):

    def act(self, X):
        return 1.0/(1.0 + np.exp(-X))
    def derivative_wrt_activation(self, act_outs):
        return act_outs*(1-act_outs)
