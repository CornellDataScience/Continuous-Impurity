import numpy as np

def affine_X(X):
    X_affine = np.ones((X.shape[0], X.shape[1]+1))
    X_affine[:,:X.shape[1]] = X
    return X_affine
