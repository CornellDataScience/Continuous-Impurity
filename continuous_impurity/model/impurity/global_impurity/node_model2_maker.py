import toolbox.data_helper as data_helper
import function.stable_func as stable_func
import numpy as np
from model.impurity.global_impurity.node_model2 import NodeModel2


def logistic_model_at_depth(X, y):
    def out(depth):
        x_shape = X.shape[1]
        def f(params_dict, k, X):
            X_affine = data_helper.affine_X(X)
            k_eq_0_out = stable_func.sigmoid(np.dot(X_affine, params_dict["theta"]))
            return k_eq_0_out if k == 0 else 1-k_eq_0_out

        def grad_f(params_dict, k, X):
            k_eq_0_out = f(params_dict, 0, X)
            X_affine = data_helper.affine_X(X)
            grad_k_eq_0_out = (k_eq_0_out*(1-k_eq_0_out))[:,np.newaxis] * X_affine
            return {"theta":grad_k_eq_0_out} if k == 0 else {"theta":-grad_k_eq_0_out}

        params_dict = {"theta": 0.000001*(np.random.rand((x_shape + 1))-0.5)}
        return NodeModel2(params_dict, f, grad_f)
    return out
