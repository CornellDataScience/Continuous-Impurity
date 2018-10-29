import numpy as np

'''contains more numerically stable forms of common functions'''
'''
From: https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
'''
def sigmoid(x):
    out = np.zeros(x.shape)
    where_greater_0 = np.where(x>=0)
    where_less_0 = np.where(x<0)
    out_greater_0 = np.exp(-x[where_greater_0])
    out_greater_0 = 1.0/(1.0 + out_greater_0)
    out_less_0 = np.exp(x[where_less_0])
    out_less_0 /= 1.0 + out_less_0
    out[where_greater_0] = out_greater_0
    out[where_less_0] = out_less_0
    return out

'''
represents tanh() in terms of the sigmoid
'''
def tanh(x):
    return 2.0*sigmoid(2.0*x) - 1.0
