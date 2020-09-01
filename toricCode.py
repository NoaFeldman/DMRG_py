from scipy import linalg
import numpy as np
import basicOperations as bops
import randomMeasurements as rm
import sys
import tensornetwork as tn
import PEPS as peps

d = 2

# Toric code model matrices - figure 30 here https://arxiv.org/pdf/1306.2164.pdf
ATensor = np.zeros((2, 2, 2, 2, 2))
ATensor[0, 0, 0, 0, 0] = 1
ATensor[1, 1, 1, 1, 0] = 1
ATensor[0, 0, 1, 1, 1] = 1
ATensor[1, 1, 0, 0, 1] = 1
A = tn.Node(ATensor, name='A', backend=None)
BTensor = np.zeros((2, 2, 2, 2, 2))
BTensor[0, 0, 0, 0, 0] = 1
BTensor[1, 1, 1, 1, 0] = 1
BTensor[0, 1, 1, 0, 1] = 1
BTensor[1, 0, 0, 1, 1] = 1
B = tn.Node(BTensor, name='B', backend=None)

AEnv = bops.permute(bops.multiContraction(A, A, '4', '4*'), [0, 4, 1, 5, 2, 6, 3, 7])
BEnv = bops.permute(bops.multiContraction(B, B, '4', '4*'), [0, 4, 1, 5, 2, 6, 3, 7])
chi = 32
# Double 'physical' leg for the closed MPS
GammaTensor = np.zeros((d, d, d, d), dtype=complex)
GammaTensor[0, 0, 0, 0] = 1
GammaTensor[0, 1, 0, 1] = 1j
GammaTensor[0, 0, 1, 1] = -1j
GammaTensor[1, 0, 1, 0] = 1j
GammaTensor[1, 1, 0, 0] = -1j
GammaTensor[1, 1, 1, 1] = 1
GammaC = tn.Node(GammaTensor / np.sqrt(3/2), name='GammaC', backend=None)
LambdaC = tn.Node(np.eye(d) / np.sqrt(d), backend=None)
GammaD = tn.Node(GammaTensor / np.sqrt(3/2), name='GammaD', backend=None)
LambdaD = tn.Node(np.eye(d) / np.sqrt(d), backend=None)

steps = 50
# cUp, cDown, dUp, dDown = peps.getBMPSRowOps(GammaC, LambdaC, GammaD, LambdaD, AEnv, BEnv, steps)
# np.save('/home/dima/PycharmProjects/DMRG_py/cUp', cUp.tensor)
# np.save('/home/dima/PycharmProjects/DMRG_py/cDown', cDown.tensor)
# np.save('/home/dima/PycharmProjects/DMRG_py/dUp', dUp.tensor)
# np.save('/home/dima/PycharmProjects/DMRG_py/dDown', dDown.tensor)

cUpTensor = np.load('/home/dima/PycharmProjects/DMRG_py/cUp.npy')
cUp = tn.Node(cUpTensor)
cDownTensor = np.load('/home/dima/PycharmProjects/DMRG_py/cDown.npy')
cDown = tn.Node(cDownTensor)
dUpTensor = np.load('/home/dima/PycharmProjects/DMRG_py/dUp.npy')
dUp = tn.Node(dUpTensor)
dDownTensor = np.load('/home/dima/PycharmProjects/DMRG_py/dDown.npy')
dDown = tn.Node(dDownTensor)

upRow = bops.multiContraction(cUp, dUp, '3', '0')
AB = bops.permute(bops.multiContraction(A, B, '3', '0'), [2, 0, 1, 4, 6, 5, 3, 7])
downRow = bops.multiContraction(cDown, dDown, '3', '0')
dmBase = bops.multiContraction(bops.multiContraction(upRow, AB, '13', '12'), AB, '12', '12*')
dmBase = bops.permute(bops.multiContraction(dmBase, downRow, [5, 11, 4, 10], '1234'), [0, 1, 2, 6, 3, 7, 10, 11, 4, 5, 8, 9])
mat = np.round(np.reshape(dmBase.tensor[1, 1, 1, 0, 1, 0, 6, 6], [4, 4]), 8)
for i in range(cUp[0].dimension):
    for k in range(d):
        for l in range(d):
            for o in range(cDown[0].dimension):
                if abs(np.trace(np.round(np.reshape(dmBase.tensor[i, i, k, l, k, l, o, o], [4, 4]), 8))) > 0:
                    m = np.round(np.reshape(dmBase.tensor[i, i, k, l, k, l, o, o], [4, 4]), 8)
                    b = 1
for i in range(cUp[0].dimension):
    for j in range(cUp[0].dimension):
        for k in range(d):
            for l in range(d):
                for m in range(d):
                    for n in range(d):
                        for o in range(cDown[0].dimension):
                            for p in range(cDown[0].dimension):
                                if abs(np.trace(np.round(np.reshape(dmBase.tensor[i, j, k, l, m, n, o, p], [4, 4]), 8))) > 0:
                                    b = 1
