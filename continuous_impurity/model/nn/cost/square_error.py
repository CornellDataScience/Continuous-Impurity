from model.nn.cost.nn_cost import NNCost
import numpy as np

class SquareError(NNCost):

    def cost(self, y, y_hat):
        y_minus_y_hat = y-y_hat
        return 0.5*np.dot(y_minus_y_hat, y_minus_y_hat)

    def d_cost(self, y, y_hat, y_hat_derivs):
        out = np.zeros(y_hat_derivs.shape[1:])
        for i in range(y_hat.shape[0]):
            out += (y[i]-y_hat[i])*y_hat_derivs[i]
        return out
