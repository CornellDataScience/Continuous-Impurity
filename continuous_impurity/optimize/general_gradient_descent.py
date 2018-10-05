import numpy as np


def sgd_minimize(f, params, param_delta, n_iter, step_size):
    params = params.copy()
    for iter in range(n_iter):
        grad = gradient(f, params, param_delta)
        print("f(params): ", f(params))
        params -= step_size * grad
    return params


def gradient(f, params, param_delta):
    gradient = np.zeros(params.shape[0])
    cost = f(params)
    for i in range(len(params)):
        original_val = params[i]
        params[i] = original_val + param_delta
        right_dcost = f(params)
        params[i] = original_val - param_delta
        left_dcost = f(params)
        params[i] = original_val
        right_slope = (right_dcost-cost)/param_delta
        left_slope = (cost - left_dcost)/param_delta
        gradient[i] = (right_slope + left_slope)/2.0
    return gradient
