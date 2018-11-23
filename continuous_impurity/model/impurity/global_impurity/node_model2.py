

class NodeModel2:

    def __init__(self, params_dict, f, grad_f):
        self._params_dict = params_dict
        self.__f = f
        self.__grad_f = grad_f

    def _f(self, child_num, X):
        return self.__f(self._params_dict, child_num, X)

    #could speed this up for functions that have gradients in terms of their inputs
    #by passing in the splits out? May not be worth the effort
    def _grad_f(self, child_num, X):
        return self.__grad_f(self._params_dict, child_num, X)
