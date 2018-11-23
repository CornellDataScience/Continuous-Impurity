import numpy as np
#is code equivalent to:
#out = np.zeros(v2.shape)
#for i in range(v1.shape[0]):
#   out[i] = v1[i]*v2[i]
#return out
#, but written to be much faster
def fast_multiply_along_first_axis(v1, v2):
    if len(v1.shape) > len(v2.shape):
        raise ValueError("v1 has greater shape than v2")
    while len(v1.shape) != len(v2.shape):
        v1 = v1[:,np.newaxis]
    return v1*v2
