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


#divides v1 by v2, and wherever v2 is zero, replaces the result with if_div_0_val
def stable_divide(v1, v2, if_div_0_val):
    #return v1/v2
    where_v2_neq_0 = np.where(v2 != 0)
    #print("where_v2_neq_0: ", (where_v2_neq_0[0]))
    out = np.full(v1.shape, if_div_0_val, dtype = v1.dtype)
    out[where_v2_neq_0] = v1[where_v2_neq_0]/v2[where_v2_neq_0]
    return out
