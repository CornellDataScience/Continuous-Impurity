import toolbox.data_helper as data_helper
import function.stable_func as stable_func
import numpy as np
from model.impurity.global_impurity3.node_model3 import NodeModel3

#assumes x_shape is the shape after x has been affined already
def logistic_model_at_depth(x_shape):

    def out(d):
        #assumes X is affined already
        def f(params, X):
            k_eq_0_out = stable_func.sigmoid(np.dot(X, params[0]))
            return np.asarray([k_eq_0_out, 1-k_eq_0_out])

        #assumes X is affined already
        def grad_f(params, X, f_outs):
            k_eq_0_out = (f_outs[0]*(1-f_outs[0]))[:,np.newaxis]*X
            return [np.array([k_eq_0_out, -k_eq_0_out])]

        params = [0.0000001*(np.random.rand((x_shape))-.5)]#do not initialize weights to zero since will cause a 0 derivative
        return NodeModel3(params, f, grad_f)
    return out
