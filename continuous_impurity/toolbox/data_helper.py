import numpy as np

def affine_X(X):
    if len(X.shape) == 1:
        out = np.ones((X.shape[0]+1), dtype = X.dtype)
        out[:out.shape[0]-1] = X
        return out
    X_affine = np.ones((X.shape[0], X.shape[1]+1), dtype = X.dtype)
    X_affine[:,:X.shape[1]] = X
    return X_affine

def unit_square_normalize(X):
    X_maxes = np.max(X, axis = 0)
    X_mins = np.min(X, axis = 0)
    if 0 in (X_maxes - X_mins):
        raise ValueError("can't unit square normalize X since at least one " + \
        "axis does not ever change value!")
    return (X-X_maxes)/(X_maxes-X_mins)

def mean_center(X):
    return X - np.average(X, axis = 0)
