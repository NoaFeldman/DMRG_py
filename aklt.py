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

UtrBase = linalg.expm(1j * np.pi * sY)
physicalUnifierTensor = bops.getLegsUnifierTensor(d, d)
physicalUnifier = tn.Node(physicalUnifierTensor, backend=None)
UtrTensor = np.tensordot(physicalUnifierTensor, UtrBase, axes=([0], [0]))
UtrTensor = np.tensordot(UtrTensor, physicalUnifierTensor, axes=([2, 0], [0, 1]))
Utr = tn.Node(UtrTensor, backend=None)

N = 32
A = [i for i in range(int(N/2))]
# A2 = [i for i in range(int(N/2), int(3 * N/4))]
psi = bops.getStartupState(N, d=d, mode='pbc')

psiP = bops.copyState(psi, conj=True)
for i in A:
    psiP[i] = bops.permute(bops.multiContraction(psiP[i], Utr, [1], [0]), [0, 2, 1])

psiDagger = bops.copyState(psi, conj=True)
psiPDagger = bops.copyState(psiP, conj=True)

temp = bops.multiContraction(psi[0], physicalUnifier, [1], [2])
tempP = bops.multiContraction(psiP[0], physicalUnifier, [1], [2])
tempD = bops.multiContraction(psiDagger[0], physicalUnifier, [1], [2])
tempPD = bops.multiContraction(psiPDagger[0], physicalUnifier, [1], [2])
curr = bops.multiContraction(bops.multiContraction(temp, tempD, [0, 2], [0, 2]),
                             bops.multiContraction(tempP, tempPD, [0, 2], [0, 2]), [1, 3], [3, 1])
# Not optimised, if D starts being big, reconsider
for i in range(1, len(A)):
    temp = bops.multiContraction(psi[i], physicalUnifier, [1], [2])
    tempP = bops.multiContraction(psiP[i], physicalUnifier, [1], [2])
    tempD = bops.multiContraction(psiDagger[i], physicalUnifier, [1], [2])
    tempPD = bops.multiContraction(psiPDagger[i], physicalUnifier, [1], [2])
    curr = bops.multiContraction(curr, bops.multiContraction(temp, tempD, [2], [2]), [0, 1], [0, 3])
    curr = bops.multiContraction(curr, bops.multiContraction(temp, tempD, [2], [2]), [0, 1, 3, 5], [0, 3, 5, 2])
curr = bops.multiContraction(bops.multiContraction(bops.multiContraction(bops.multiContraction(curr, psi[i+1], [0], [0]), psiDagger[i+1],
                                                                  [0, 3, 4], [0, 1, 2]), psiP[i+1], [0], [0]), psiPDagger[i+1], [0, 1, 2], [0, 1, 2])
b = 1
