import numpy as np
import pickle
import basicOperations as bops
from matplotlib import pyplot as plt

with open('problematicMatrix', 'rb') as f:
    node = pickle.load(f)
leftEdges = [0, 1, 2]
rightEdges = [3, 4, 5]
mat = np.reshape(node.tensor, [np.prod([np.shape(node.tensor)[i] for i in leftEdges]), np.prod([np.shape(node.tensor)[i] for i in rightEdges])])
print(np.linalg.cond(mat))
mat = np.round(mat, 16)
print(np.linalg.cond(mat))
[u, s, vh] = np.linalg.svd(mat)
matMatDagger = np.matmul(mat, np.conj(np.transpose(mat)))
[s2, myu_] = np.linalg.eigh(matMatDagger)
isMat = np.matmul(myu_, np.matmul(s2, np.conj(np.transpose(myu_))))
umat = np.matmul(np.conj(np.transpose(myu_)), mat)
[myu, mys, myvh] = np.linalg.svd(umat)
