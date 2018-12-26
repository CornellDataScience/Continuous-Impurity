from abc import ABC, abstractmethod

class NNCost(ABC):
    #allows for scalar/vector y and y_hat, and returns the cost for an estimate of y, y_hat
    @abstractmethod
    def cost(self, y, y_hat):
        pass

    #returns the cost gradient given y, y_hat (estimate for y), and the derivatives of
    #y_hat w.r.t. some parameter. y_hat_derivs has shape(len(y_hat),) + param_shape
    @abstractmethod
    def d_cost(self, y, y_hat, y_hat_derivs):
        pass
