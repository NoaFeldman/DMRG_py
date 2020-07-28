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

# sXsX = np.kron(sX, sX)
# sYsY = np.kron(sY, sY)
# sZsZ = np.kron(sZ, sZ)


"""A Random matrix distributed with Haar measure"""
def haar_measure(n):
    z = (np.random.randn(n, n) + 1j * np.random.randn(n, n)) / np.sqrt(2.0)
    q,r = scipy.linalg.qr(z)
    d = np.diagonal(r)
    ph = d / np.absolute(d)
    q = np.multiply(q, ph, q)
    return q


UtrBase = linalg.expm(1j * np.pi * sY)
physicalUnifierTensor = bops.getLegsUnifierTensor(d, d)
physicalUnifier = tn.Node(physicalUnifierTensor, backend=None)
UtrTensor = np.tensordot(physicalUnifierTensor, UtrBase, axes=([0], [0]))
UtrTensor = np.tensordot(UtrTensor, physicalUnifierTensor, axes=([2, 0], [0, 1]))
Utr = tn.Node(UtrTensor, backend=None)


def getRandomUnitary():
    u1 = haar_measure(d)
    u2 = haar_measure(d)
    physicalUnifier = tn.Node(physicalUnifierTensor, backend=None)
    UTensor = np.tensordot(physicalUnifierTensor, u1, axes=([0], [0]))
    UTensor = np.tensordot(UTensor, u2, axes=([0], [0]))
    UTensor = np.tensordot(UTensor, physicalUnifierTensor, axes=([1, 2], [0, 1]))
    return tn.Node(UTensor, backend=None), u1, u2


N = 32
A = [i for i in range(int(N/2))]
# A2 = [i for i in range(int(N/2), int(3 * N/4))]
psi = bops.getStartupState(N, d=d, mode='pbc')

psiP = bops.copyState(psi, conj=True)
for i in A:
    psiP[i] = bops.permute(bops.multiContraction(psiP[i], Utr, [1], [0]), [0, 2, 1])

M = 5000
for m in range(M):
    psiCopy = bops.copyState(psi)
    psiPCopy = bops.copyState(psiP)
    unitaries = [[None, None] for i in A]
    unitariesP = [[None, None] for i in A]
    for i in A:
        U, u1, u2 = getRandomUnitary()
        Up, u1p, u2p = getRandomUnitary()
        unitaries[i] = [u1, u2]
        unitariesP[i] = [u1p, u2p]
        psiCopy[i] = bops.permute(bops.multiContraction(psiCopy[i], U, [1], [0]), [0, 2, 1])
        psiPCopy[i] = bops.permute(bops.multiContraction(psiPCopy[i], Up, [1], [0]), [0, 2, 1])
        

