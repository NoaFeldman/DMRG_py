import numpy as np

def stepMat(E, W):
    mat = np.zeros(2)
    mat[0, 1] = 1
    mat[1, 0] = -1
    mat[1, 1] = E - np.random.uniform(-W/2, W/2)
    return mat

