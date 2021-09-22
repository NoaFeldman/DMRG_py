import numpy as np
import matplotlib.pyplot as plt

def tmat(W, E):
    mat = np.zeros((2, 2))
    mat[0, 1] = 1
    mat[1, 0] = -1
    mat[1, 1] = E - np.random.uniform(-W/2, W/2)
    return mat

E = 0
W = 1
L = 10000
T = np.eye(2)
xisG = np.zeros(L)
np.seterr(divide='raise')
scaleFixer = 0
for l in range(L):
    T = np.matmul(T, tmat(W, E))
    w = np.linalg.eigvals(T)
    if np.real(w[0]) != w[0]:
        b=1
    lambdaG = max(np.abs(w))
    try:
        xisG[l] = l / (np.log(lambdaG) + scaleFixer)
    except:
        pass
    T /= lambdaG
    scaleFixer += np.log(lambdaG)
plt.plot(xisG[1500:])
# plt.plot(xisL[1500:])
plt.show()