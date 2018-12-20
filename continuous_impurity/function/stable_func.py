import numpy as np
from scipy.special import expit
'''contains more numerically stable forms of common functions'''
'''
From: https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
'''
def sigmoid(x):
    return expit(x)

'''
represents tanh() in terms of the sigmoid
'''
def tanh(x):
    return 2.0*sigmoid(2.0*x) - 1.0
