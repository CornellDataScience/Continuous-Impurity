import numpy as np

def plot_contours(X, f, ax, step_size, **params):
    xx, yy = make_meshgrid(X[:,0], X[:,1], h = step_size)
    Xs = np.c_[xx.ravel(), yy.ravel()]
    f_outs = f(Xs)#np.zeros(Xs.shape[0])
    #for i in range(0, len(f_outs)):
    #    f_outs[i] = f(Xs[i])
    f_outs = f_outs.reshape(xx.shape)
    out = ax.contourf(xx, yy, f_outs, **params)
    return out


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy
