import numpy as np

class FeedForwardNN:

    #assumes layer_lengths[0] is the length of x
    def __init__(self, layer_lengths, act_funcs, cost):
        assert(len(act_funcs) == len(layer_lengths)-1)
        self.__init_params(layer_lengths)
        self.__act_funcs = act_funcs
        self.__cost = cost

    def __init_params(self, layer_lengths):
        self._A = []
        self._b = []
        for l in range(len(layer_lengths) - 1):
            rows = layer_lengths[l+1]
            cols = layer_lengths[l]
            self._A.append(self.__rand_init_weights(rows, cols))
            self._b.append(self.__rand_init_biases(rows))

    def __rand_init_weights(self, rows, cols):
        return 0.0000000001*(np.random.rand(rows, cols)-.5)

    def __rand_init_biases(self, n_biases):
        return 0.00000001*(np.random.rand(n_biases)-.5)

    def forward(self, x):
        last_layer_out = x
        out = [x]
        for l in range(len(self._A)):
            A_l = self._A[l]
            b_l = self._b[l]
            act_l = self.__act_funcs[l].act
            this_layer_out = act_l(np.dot(A_l, last_layer_out.T).T + b_l)
            out.append(this_layer_out)
            last_layer_out = this_layer_out
        return out

    def backward(self, forwards):
        x = forwards[0]
        forwards = forwards[1:]
        d_forwards = [self.__act_funcs[i].derivative_wrt_activation(forwards[i]) for i in range(0,len(forwards))]
        A_grad = [np.zeros((forwards[-1].shape[0],) + A.shape, dtype = A.dtype) for A in self._A]
        b_grad = [np.zeros((forwards[-1].shape[0],) + b.shape, dtype = b.dtype) for b in self._b]

        self.__set_last_layer_grads(forwards, d_forwards, A_grad, b_grad)
        P_q = None
        for q in range(len(A_grad)-2, -1, -1):
            A_parent = self._A[q+1]
            D_parent = np.zeros((d_forwards[q+1].shape[0], d_forwards[q+1].shape[0]))
            for i in range(D_parent.shape[0]):
                D_parent[i,i] = d_forwards[q+1][i]
            P_q = np.dot(D_parent, A_parent) if P_q is None else np.dot(P_q, np.dot(D_parent, A_parent))

            forwards_prev = x if q == 0 else forwards[q-1]
            for m in range(A_grad[q].shape[1]):
                for n in range(A_grad[q].shape[2]):
                    A_dot_vec = np.zeros(P_q.shape[1])
                    A_dot_vec[m] = d_forwards[q][m]*forwards_prev[n]
                    A_grad[q][:,m,n] = np.dot(P_q, A_dot_vec)
                b_dot_vec = np.zeros(P_q.shape[1])
                b_dot_vec[m] = d_forwards[q][m]
                b_grad[q][:,m] = np.dot(P_q, b_dot_vec)
        return A_grad, b_grad


    def cost_grad(self, x, y):
        forwards = self.forward(x)
        y_hat = forwards[-1]
        y_hat_A_grad, y_hat_b_grad = self.backward(forwards)
        cost_A_grad = [None for i in range(len(y_hat_A_grad))]
        cost_b_grad = [None for i in range(len(y_hat_b_grad))]
        for l in range(0, len(cost_A_grad)):
            cost_A_grad[l] = self.__cost.d_cost(y, y_hat, y_hat_A_grad[l])
            cost_b_grad[l] = self.__cost.d_cost(y, y_hat, y_hat_b_grad[l])
        return cost_A_grad, cost_b_grad




    def __set_last_layer_grads(self, forwards, d_forwards, A_grad, b_grad):
        l = len(A_grad)-1
        for i in range(A_grad[l].shape[0]):
            for m in range(A_grad[l].shape[1]):
                if i == m:
                    b_grad[l][m] = d_forwards[l][i]

                    for n in range(A_grad[l].shape[2]):
                        A_grad[l][i,m,n] = d_forwards[l][i] * forwards[l-1][n]
