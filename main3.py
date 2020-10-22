import scipy
from scipy import linalg
import tensornetwork as tn
import numpy as np
import basicOperations as bops
import DMRG as dmrg
from matplotlib import pyplot as plt
from partialTranspose import getPartiallyTransposed, getStraightDM

d = 3

sX = np.zeros((d, d), dtype=complex)
sX[0, 1] = 1 / np.sqrt(2)
sX[1, 0] = 1 / np.sqrt(2)
sX[1, 2] = 1 / np.sqrt(2)
sX[2, 1] = 1 / np.sqrt(2)
sZ = np.zeros((d, d), dtype=complex)
sZ[0, 0] = -1
sZ[1, 1] = 0
sZ[2, 2] = 1
sY = 1j * (np.matmul(sZ, sX) - np.matmul(sX, sZ))

sXsX = np.kron(sX, sX)
sYsY = np.kron(sY, sY)
sZsZ = np.kron(sZ, sZ)

Utr = linalg.expm(1j * np.pi * sY)

def getAKLTHamiltonianMatrices(N):
    onsiteTerms = [np.zeros((d, d))] * N
    neighborTerms = [sXsX + sYsY + sZsZ + np.matmul(sXsX + sYsY + sZsZ, sXsX + sYsY + sZsZ) / 3] * N
    return onsiteTerms, neighborTerms


def getHamiltonianMatrices(N, B, D):
    onsiteTerms = [B * sX + D * np.matmul(sZ, sZ)] * N
    neighborTerms = [sXsX + sYsY + sZsZ] * N
    return onsiteTerms, neighborTerms


"""A Random matrix distributed with Haar measure"""
def haar_measure(n):
    z = (np.random.randn(n, n) + 1j * np.random.randn(n, n)) / np.sqrt(2.0)
    q,r = scipy.linalg.qr(z)
    d = np.diagonal(r)
    ph = d / np.absolute(d)
    q = np.multiply(q, ph, q)
    return q


def randomUnitary(site):
    return tn.Node(haar_measure(d), name=('U' + str(site)),
                             axis_names=['s' + str(site), 's' + str(site) + '*'],
                             backend=None)


def trUnitary(site):
    return tn.Node(Utr, name=('U' + str(site)),
                             axis_names=['s' + str(site), 's' + str(site) + '*'],
                             backend=None)


N = 32
B = 0
D = 0
A1 = [i for i in range(int(N/4), int(N/2))]
A2 = [i for i in range(int(N/2), int(3 * N/4))]
onsiteTerms, neighborTerms = getHamiltonianMatrices(N, B, D)
psi = bops.getStartupState(N, d=d)
# H = dmrg.getDMRGH(N, onsiteTerms, neighborTerms, d=3)
# HLs, HRs = dmrg.getHLRs(H, psi)
# psi, E0, truncErrs = dmrg.getGroundState(H, HLs, HRs, psi, None)
# print('E0 = ' + str(E0))
for k in [len(psi) - 1 - i for i in range(len(psi) - A2[-1] - 1)]:
    psi = bops.shiftWorkingSite(psi, k, '<<')

# psiDagger = bops.copyState(psi, conj=True)
# i = A1[-1]
# sigma = trUnitary(i)
# temp = bops.copyState([psi[i]])[0]
# temp.tensor = np.conj(temp.tensor)
# T = bops.permute(bops.multiContraction(bops.multiContraction(temp, sigma, [1], [0]), psiDagger[i], [2], [1]), [0, 2, 1, 3])
# origSizes = [T.tensor.shape[0], T.tensor.shape[1]]
# tMatrix = np.round(np.reshape(T.tensor, [origSizes[0] * origSizes[1], origSizes[0] * origSizes[1]]), 14)
# vals, vecs = scipy.sparse.linalg.eigs(tMatrix, k=1)
# # vals = np.round(vals, 6)
# # uVec = vecs[:, list(vals).index(1)]
# uVec = vecs[:, 0]
# uMatrix = np.reshape(uVec, origSizes)
# uMatrix /= np.sqrt(np.trace(np.matmul(uMatrix, np.conj(np.transpose(uMatrix)))) / len(uMatrix))
# print(np.trace(np.matmul(uMatrix, np.conj(uMatrix))) / len(uMatrix))
# U = tn.Node(uMatrix, backend=None)


psiP = bops.copyState(psi, conj=True)
for i in A1 + A2:
    currU = trUnitary(i)
    psiP[i] = bops.permute(bops.multiContraction(psiP[i], currU, [1], [0]), [0, 2, 1])
psiConj = bops.copyState(psi, conj=True)
curr = bops.multiContraction(psiP[A1[0]], psiConj[A1[0]], [0, 1], [0, 1])
for i in range(A1[1], A2[-1]-1):
    curr = bops.multiContraction(bops.multiContraction(curr, psiP[i], [0], [0]), psiConj[i], [0, 1], [0, 1])
curr = bops.multiContraction(bops.multiContraction(curr, psiP[A2[-1]], [0], [0]), psiConj[A2[-1]], [0, 1, 2], [0, 1, 2])
# pt = getPartiallyTransposed(psi, A1[0], A1[-1], A2[0], A2[-1])
# reg = getStraightDM(psiP, A1[0], A1[-1], A2[0], A2[-1])
b=1
""" This does not give the same result as Poleman-Turner's paper! """

def applySwap(psi, psiP, startInd, endInd):
    psiDagger = bops.copyState(psi, conj=True)
    psiPDagger = bops.copyState(psi, conj=True)
    curr = bops.multiContraction(psi[startInd], psiDagger[startInd], [0], [0])
    currP = bops.multiContraction(psiP[startInd], psiPDagger[startInd], [0], [0])
    curr = bops.multiContraction(curr, currP, [2, 0], [0, 2])
    for i in range(startInd + 1, endInd + 1):
        curr = bops.multiContraction(curr, psi[i], [0], [0])
        curr = bops.multiContraction(curr, psiDagger[i], [0], [0])
        curr = bops.multiContraction(curr, psiP[i], [0, 4], [0, 1])
        curr = bops.multiContraction(curr, psiPDagger[i], [0, 1], [0, 1])
    bops.removeState(psiPDagger)
    bops.removeState(psiPDagger)
    return curr


def applyO(psi, psiP, startInd, endInd):
    psiDagger = bops.copyState(psi, conj=True)
    psiPDagger = bops.copyState(psi, conj=True)
    OTensor = np.zeros((9, 9), dtype=complex)
    for i in range(3):
        for j in range(3):
            OTensor[i * 3 + i, j * 3 + j] = 1
    OTensor = np.reshape(OTensor, [3, 3, 3, 3])
    O = tn.Node(OTensor, name='O', backend=None)
    curr = bops.multiContraction(psi[endInd], psiDagger[endInd], [2], [2])
    currP = bops.multiContraction(psiP[endInd], psiPDagger[endInd], [2], [2])
    curr = bops.multiContraction(curr, O, [1, 3], [1, 3])
    curr = bops.multiContraction(curr, currP, [2, 3], [1, 3])
    for i in [endInd - j for j in range(1, endInd - startInd + 1)]:
        O = tn.Node(OTensor, name='O', backend=None)
        curr = bops.multiContraction(curr, psi[i], [0], [2])
        curr = bops.multiContraction(curr, psiDagger[i], [0], [2])
        curr = bops.multiContraction(curr, O, [3, 5], [1, 3])
        curr = bops.multiContraction(curr, psiP[i], [0, 4], [2, 1])
        curr = bops.multiContraction(curr, psiPDagger[i], [0, 3], [2, 1])
    bops.removeState(psiPDagger)
    bops.removeState(psiPDagger)
    return curr


M = 100
mySamples = [None for m in range(M)]
for m in range(M):
    psiCopy = bops.copyState(psi)
    for i in A1 + A2:
        U = randomUnitary(i)
        psiCopy[i][1] ^ U[1]
        psiCopy[i] = bops.permute(tn.contract_between(psiCopy[i], U), [0, 2, 1])
    psiPCopy = bops.copyState(psiP)
    for i in A1 + A2:
        U = randomUnitary(i)
        psiPCopy[i][1] ^ U[1]
        psiPCopy[i] = bops.permute(tn.contract_between(psiPCopy[i], U), [0, 2, 1])
    swapped = applySwap(psiCopy, psiPCopy, A1[0], A1[-1])
    Oed = applyO(psi, psiP, A2[0], A2[-1])
    mySamples[m] = bops.multiContraction(swapped, Oed, [0, 1, 2, 3], [0, 1, 2, 3]).tensor
    bops.removeState(psiCopy)
    bops.removeState(psiPCopy)

b = 1


