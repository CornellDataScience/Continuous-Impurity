

class NodeModel3:

    #- params is a list of params
    #- f is of the form: f: params -> X -> A, where A[k][i] = f(k|X[i])
    #- grad_f is of the form: grad_f: params -> X -> f_outs, where f_outs[k][i] = f(k|X[i]).
    #  Must return A s.t. A[k_id, param_num, k_child_id, i] = grad f_k(k_child|X[i]) w.r.t. param (param_num)
    def __init__(self, params, f, grad_f):
        self._params = params
        self.__f = f
        self.__grad_f = grad_f

    def _f(self, X):
        return self.__f(self._params, X)


    def _grad_f(self, X, f_outs):
        return self.__grad_f(self._params, X, f_outs)
