from abc import ABC, abstractmethod

class ActivationFunction(ABC):
    @abstractmethod
    def act(self, X):
        pass

    @abstractmethod
    def derivative_wrt_activation(self, act_outs):
        pass
