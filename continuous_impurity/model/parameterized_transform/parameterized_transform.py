from abc import ABC, abstractmethod

class ParameterizedTransform(ABC):


    def __init__(self, params):
        #ideally this would be more private, but lots of models pipeline gradients
        #through the transformation gradient, requiring an easy way to
        #step the params. Otherwise would just end up implementing
        #both a getter and setter which at that point it may as well
        #just be public
        self.params = params

    '''
    Transforms X, where X is a matrix whose rows are the vectors to be transformed.
    '''
    @abstractmethod
    def transform(self, X):
        pass

    '''
    Returns a matrix, A, with shape transform(X).shape + params.shape
    s.t. A[i,j,k,l] is partial (T(X[i]))[j]/ partial params[k,l]
    '''
    @abstractmethod
    def param_grad(self, X, transform_outs):
        pass
