import numpy as np
import basicOperations as bops
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
GammaTensor[1, 0, 0, 1] = -1
GammaTensor[0, 1, 0, 1] = 1j
GammaTensor[1, 1, 0, 0] = -1j
GammaTensor[0, 0, 1, 1] = -1j
GammaTensor[1, 0, 1, 0] = 1j
GammaTensor[0, 1, 1, 0] = 1
GammaTensor[1, 1, 1, 1] = -1
GammaC = tn.Node(GammaTensor / np.sqrt(2), name='GammaC', backend=None)
LambdaC = tn.Node(np.eye(d) / np.sqrt(d), backend=None)
GammaD = tn.Node(GammaTensor / np.sqrt(2), name='GammaD', backend=None)
LambdaD = tn.Node(np.eye(d) / np.sqrt(d), backend=None)

peps.checkCannonization(GammaC, LambdaC, GammaD, LambdaD)

steps = 50
cUp, cDown, dUp, dDown = peps.getBMPSRowOps(GammaC, LambdaC, GammaD, LambdaD, AEnv, BEnv, steps)
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
dmBase = bops.permute(bops.multiContraction(dmBase, downRow, [5, 11, 4, 10], '1234'),
                      [0, 1, 2, 6, 3, 7, 10, 11, 4, 5, 8, 9])
# for i in range(cUp[0].dimension):
#     for j in range(cUp[0].dimension):
#         for k in range(d):
#             for l in range(d):
#                 for m in range(d):
#                     for n in range(d):
#                         for o in range(dDown[3].dimension):
#                             for p in range(dDown[3].dimension):
#                                 mat = np.round(np.reshape(dmBase.tensor[i, j, k, l, m, n, o, p], [4, 4]), 8)
#                                 if mat[0, 0] > 0.1:
#                                     b = 1
#                                 if mat[1, 1] > 0.1:
#                                     b = 1
#                                 if mat[2, 2] > 0.1:
#                                     b = 1
#                                 if mat[3, 3] > 0.1:
#                                     b = 1

# Calculation in depol_scribbles
x = np.real(np.reshape(dmBase.tensor[0, 0, 0, 0, 0, 0, 2, 2], [4, 4])[0, 0])
y = np.real(np.reshape(dmBase.tensor[0, 0, 0, 0, 0, 0, 2, 2], [4, 4])[2, 2])
z = np.real(np.reshape(dmBase.tensor[0, 0, 0, 0, 1, 1, 0, 1], [4, 4])[1, 1])
w = np.imag(np.reshape(dmBase.tensor[0, 0, 0, 0, 1, 1, 0, 1], [4, 4])[1, 1])
q = np.real(np.reshape(dmBase.tensor[0, 0, 0, 0, 1, 1, 0, 1], [4, 4])[3, 3])
r = np.imag(np.reshape(dmBase.tensor[0, 0, 0, 0, 1, 1, 0, 1], [4, 4])[3, 3])

tempMat = np.array([[z + q, w + r, z + q], [w, q, -r], [r, z, -w]])
tempV = np.array([1 - x - y, 0, 0])
gamma, phi, epsilon = np.matmul(np.linalg.inv(tempMat), tempV)[:]
delta = 0
alpha = 1
mu = 1 - alpha
beta = 0
nu = -beta

XbTensor = np.zeros((cDown[0].dimension, d, d, cUp[0].dimension), dtype=complex)
XbTensor[2, 0, 0, 0] = alpha + 1j * beta
XbTensor[3, 1, 1, 0] = mu + 1j * nu
XbTensor[0, 0, 0, 0] = gamma + 1j * delta
XbTensor[1, 1, 1, 0] = epsilon + 1j * phi
Xb = tn.Node(XbTensor)
XaTensor = np.zeros((dUp[3].dimension, d, d, dDown[3].dimension), dtype=complex)
XaTensor[0, 0, 0, 2] = 1
XaTensor[0, 1, 1, 3] = 1
XaTensor[0, 1, 1, 1] = 1
XaTensor[0, 0, 0, 0] = 1
Xa = tn.Node(XaTensor)

dm = bops.multiContraction(bops.multiContraction(Xb, dmBase, '0123', '6230'), Xa, '0123', '0123')
m = np.round(np.reshape(dm.tensor, [4, 4]), 8)

Xa, Xb = peps.getBMPSSiteOps(50, Xa, Xb, cUp, cDown, dUp, dDown, AEnv, BEnv)

b = 1

