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
""" from https://arxiv.org/pdf/math-ph/0609050.pdf """
def haar_measure(n):
    z = (np.random.randn(n, n) + 1j * np.random.randn(n, n)) / np.sqrt(2.0)
    q, r = scipy.linalg.qr(z)
    d = np.diagonal(r)
    ph = d / np.absolute(d)
    q = np.multiply(q, ph, q)
    return q


# O = np.zeros((d**2, d**2), dtype=complex)
# O[2, 2] = 4
# O[6, 2] = 1
# O[2, 6] = 1
# O[1, 3] = 1
# O[3, 1] = 1
# swap = np.zeros((d**2, d**2), dtype=complex)
# for i in range(d):
#     for j in range(d):
#         swap[i * d + j, j * d + i] = 1
# M = 10000
# mysum = np.zeros((d**2, d**2), dtype=complex)
# for m in range(M):
#     u1 = haar_measure(d)
#     # u2 = haar_measure(d)
#     U = np.kron(u1, u1)
#     # U = np.kron(np.kron(u1, u2), np.kron(u1, u2))
#     phi = np.matmul(np.matmul(np.conj(np.transpose(U)), O), U)
#     mysum += phi
# avg = mysum / M
# expected = 1 / (d**2 - 1) * (np.eye(d**2) * (d**2 - 1) + swap * (-d + np.trace(np.matmul(swap, O))))
# b = 1

UtrBase = linalg.expm(1j * np.pi * sY)
physicalUnifierTensor = bops.getLegsUnifierTensor(d, d)
physicalUnifier = tn.Node(physicalUnifierTensor, backend=None)
UtrTensor = np.tensordot(physicalUnifierTensor, UtrBase, axes=([0], [0]))
UtrTensor = np.tensordot(UtrTensor, physicalUnifierTensor, axes=([2, 0], [0, 1]))
Utr = tn.Node(UtrTensor, backend=None)


def getRandomUnitary():
    u1 = haar_measure(d)
    """ Note this debugging change! """
    # u2 = haar_measure(d)
    u2 = np.eye(d)
    physicalUnifier = tn.Node(physicalUnifierTensor, backend=None)
    UTensor = np.tensordot(physicalUnifierTensor, u1, axes=([0], [0]))
    UTensor = np.tensordot(UTensor, u2, axes=([0], [0]))
    UTensor = np.tensordot(UTensor, physicalUnifierTensor, axes=([1, 2], [0, 1]))
    return tn.Node(UTensor, backend=None), u1, u2


def getDistance(s, sP, N):
    sExplicit = [0] * N
    sPExplicit = [0] * N
    for i in range(N):
        sExplicit[i] = s // d**(N - 1 - i)
        sPExplicit[i] = sP // d**(N - 1 - i)
        s = s % d**(N - 1 - i)
        sP = sP % d**(N - 1 - i)
    return N - sum(np.array(sPExplicit) == np.array(sExplicit))


baseTensor = np.zeros((2, 3, 2), dtype=complex)
baseTensor[0, 0, 1] = np.sqrt(2 / 3)
baseTensor[0, 1, 0] = -np.sqrt(1 / 3)
baseTensor[1, 1, 1] = np.sqrt(1 / 3)
baseTensor[1, 2, 0] = -np.sqrt(2 / 3)
base = [tn.Node(baseTensor, axis_names=['v0', 's0', 'v1'], backend=None),
        tn.Node(np.copy(baseTensor), axis_names=['v1', 's1', 'v2'], backend=None)]
base[0].tensor /= np.sqrt(bops.getOverlap(base, base))
couple = bops.multiContraction(base[0], base[1], [2], [0])
expected = np.reshape(bops.multiContraction(couple, bops.copyState([couple], conj=True)[0], [0, 3], [0, 3]).tensor, [d**2, d**2])
M = 10000
mysum = np.zeros((d**2, d**2), dtype=complex)
for m in range(M):
    curr = bops.copyState(base)
    us = [None, None]
    for i in range(len(base)):
        uTensor = haar_measure(d)
        U = tn.Node(uTensor, axis_names=['s' + str(i), 's' + str(i) + '*'], backend=None)
        curr[i] = bops.permute(bops.multiContraction(curr[i], U, [1], [0]), [0, 2, 1])
        us[i] = uTensor
    u = np.kron(us[0], us[1])
    couple = bops.multiContraction(curr[0], curr[1], [2], [0])
    rho = np.reshape(bops.multiContraction(couple, bops.copyState([couple], conj=True)[0], [0, 3], [0, 3]).tensor, [d**2, d**2])
    for s in range(d**2):
        for sP in range(d**2):
            projector = np.zeros((d**2, d**2))
            projector[sP, sP] = 1
            mysum += (-d)**getDistance(s, sP, 2) * rho[s, s] * \
                     np.matmul(np.matmul(np.conj(np.transpose(u)), projector), u)
    b = 1
avg = mysum / M

N = 32
A = [i for i in range(1)]
# A2 = [i for i in range(int(N/2), int(3 * N/4))]
psi = bops.getStartupState(N, d=d, mode='pbc')

psiP = bops.copyState(psi, conj=True)
for i in A:
    psiP[i] = bops.permute(bops.multiContraction(psiP[i], Utr, [1], [0]), [0, 2, 1])


def getProjector(state, spaceSize):
    m = np.zeros((spaceSize, spaceSize))
    m[state, state] = 1
    return tn.Node(m, backend=None)


def measureInComputationalBasis(psi):
    psiCopy = bops.copyState(psi)
    res = [None for i in A]
    for i in A:
        randomMeas = np.random.uniform()
        probabilitiesCovered = 0
        for s in range(d**2):
            projector = getProjector(s, d**2)
            sCopy = bops.copyState(psiCopy)
            sCopy[i] = bops.permute(bops.multiContraction(sCopy[i], projector, [1], [0]), [0, 2, 1])
            currProbability = bops.getOverlap(psiCopy, sCopy)
            if randomMeas >= probabilitiesCovered and randomMeas <= probabilitiesCovered + currProbability:
                res[i] = s
                bops.removeState(psiCopy)
                psiCopy = sCopy
                norm = bops.getOverlap(psiCopy, psiCopy)
                psiCopy[0] = bops.multNode(psiCopy[0], 1 / np.sqrt(norm))
                break
            probabilitiesCovered += currProbability
            bops.removeState(sCopy)
    finalRes = [None for i in range(len(res) * 2)]
    for i in range(len(res)):
        finalRes[i] = int(res[i] / d)
        finalRes[len(finalRes) - 1 - i] = res[i] % d
    bops.removeState(psiCopy)
    return finalRes


def estimateFromMeasurement(string, unitaries):
    res = [None for i in range(len(string))]
    for i in range(len(string)):
        if i < int(len(string) / 2):
            u = unitaries[i][0]
        else:
            u = unitaries[len(string) - 1 - i][1]
        res[i] = 4 * np.matmul(np.matmul(np.conj(np.transpose(u)), getProjector(string[i], d).tensor), u) - np.eye(d)
    return res


M = 5000
sums = [np.zeros((d, d), dtype=complex) for i in range(len(A)*2)]
sumsP = [np.zeros((d, d), dtype=complex) for i in range(len(A)*2)]
for m in range(M):
    psiCopy = bops.copyState(psi)
    psiPCopy = bops.copyState(psiP)
    unitaries = [None for i in range(len(psi))]
    unitariesP = [None for i in range(len(psi))]
    for i in A:
        U, u1, u2 = getRandomUnitary()
        unitaries[i] = [u1, u2]
        Up, u1p, u2p = getRandomUnitary()
        unitariesP[i] = [u1p, u2p]
        psiCopy[i] = bops.permute(bops.multiContraction(psiCopy[i], U, [1], [0]), [0, 2, 1])
        psiPCopy[i] = bops.permute(bops.multiContraction(psiPCopy[i], Up, [1], [0]), [0, 2, 1])
    cBasisRes = measureInComputationalBasis(psiCopy)
    estimatedRho = estimateFromMeasurement(cBasisRes, unitaries)
    cBasisResP = measureInComputationalBasis(psiPCopy)
    estimatedRhoP = estimateFromMeasurement(cBasisResP, unitariesP)
    for i in range(len(A)*2):
        sums[i] += estimatedRho[i]
        sumsP[i] += estimatedRhoP[i]
avgs = [mat / M for mat in sums]
avgsP = [mat / M for mat in sumsP]
b = 1


