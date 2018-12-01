import numpy as np


def create_rect_simple(rect_region = np.array([[-.5,.5],[-.5,.5]]),step_size = 0.25):
    xs = np.arange(-1,1,step_size)
    ys = np.arange(-1,1,step_size)
    print("xs: ", xs)
    X = np.zeros((xs.shape[0]*ys.shape[0],2))
    y_out = np.zeros(X.shape[0], dtype = np.int)
    i = 0
    for x in xs:
        for y in ys:
            X[i] = np.array([x,y])
            if x >= rect_region[0,0] and x <= rect_region[0,1] and y >= rect_region[1,0] and y <= rect_region[1,1]:
                y_out[i] = 1
            else:
                y_out[i] = 0
            i+=1
    return X,y_out



'''
TODO:
A synthetic dataset construcor that fills the entire space of dims and
adds cubic regions for labels.

assumes all elements in cube_locs are of dims dimensions
'''
def create_cubes(dims, cube_locs = None, cube_labels = None, step_size = 0.01):
    return None
