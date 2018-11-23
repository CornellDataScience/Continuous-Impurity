

class NodeModel2:

    '''
    - where params_dict is a dictionary whose keys
        are the names of the individual paramaters
    - where func is of the form func (params_dict, k, X) -> (A, X_transformed)
        - where A is a matrix s.t. A[i(ind in X)]
          is p(k|X[i]),
        - and X_transformed is what input of X should be used as input for child k
    - where grad_func of the form grad_func(params_dict, k, X) -> (dict, X_transformed dict)
        - where dict[key] where key is a key in params_dict is A s.t.
          A[i(ind in X)] is grad w.r.t. params_dict[key] p(k|X[i]) (thus A[i] has
          the same shape as params_dict[key]). If params[key] does not contribute at all
          to func[0](params_dict, k, X), dict[key] is None.
        - where X_transformed_dict[key] where key is a key in params_dict is B s.t.
          B[i(ind in X)] is grad w.r.t. params_dict[key] func[1](params_dict, k, X) (thus A[i]
          has the same shape as params_dict[key]). If params[key] does not contribute at all
          to func[1](params_dict, k, X), dict[key] is None.
    '''
    def __init__(self, params_dict, func, grad_func):
        self.__params_dict = params_dict
        self.__func = func
        self.__grad_func = grad_func

    def _func(self, k, X):
        return self.__func(self.__params_dict, k, X)

    def _grad_func(self, k, X):
        return self.__grad_func(self.__params_dict, k, X)
