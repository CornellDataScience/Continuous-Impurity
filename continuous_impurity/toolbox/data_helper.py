import numpy as np

def train_test_split(X, y, percent_train_samples, seed = None):
    num_train_samples = int(percent_train_samples*X.shape[0])
    if seed is not None:
        np.random.seed(seed)
    permute_inds = np.random.permutation(np.arange(0, X.shape[0]))

    train_inds = permute_inds[:num_train_samples]
    test_inds = permute_inds[num_train_samples:]
    X_train = X[train_inds]
    y_train = y[train_inds]
    X_test = X[test_inds]
    y_test = y[test_inds]
    return (X_train, y_train), (X_test, y_test)



#assumes model(X) returns y^, predictions for y
def evaluate_accuracy(model, X, y):
    predictions = model.predict(X)
    return float(np.sum(predictions == y))/float(X.shape[0])


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
