import numpy as np
import basicOperations as bops
import tensornetwork as tn
import magic.basicDefs as basicdefs
from typing import List
import magic.magicRenyi as renyi
import DMRG as dmrg
import matplotlib.pyplot as plt
import pickle

digs = '012'
def int2base(x, base=3, length=None):
    if x == 0:
        res = '0'
    digits = []
    while x:
        digits.append(digs[int(x % base)])
        x = int(x / base)
    digits.reverse()
    res = ''.join(digits)
    if length is None:
        return res
    return '0' * (length - len(res)) + res


# def getAKLTState(N):
#     sigmaX = np.zeros((2, 2), dtype=complex)
#     sigmaX[1, 0] = 1
#     sigmaX[0, 1] = 1
#     sigmaY = np.zeros((2, 2), dtype=complex)
#     sigmaY[0, 1] = -1j
#     sigmaY[1, 0] = 1j
#     sigmaPlus = 0.5 * (sigmaX + sigmaY)
#     sigmaMinus = 0.5 * (sigmaX - sigmaY)
#     sigmaZ = np.eye(2, dtype=complex)
#     sigmaZ[1, 1] = -1
#     baseTensor = np.zeros((2, 3, 2), dtype=complex)
#     baseTensor[:, 0, :] = sigmaPlus * np.sqrt(2)
#     baseTensor[:, 1, :] = -sigmaMinus * np.sqrt(2)
#     baseTensor[:, 2, :] = sigmaZ
#
#     leftVec = np.array([1, 0])
#     leftTensor = np.tensordot(leftVec, baseTensor, axes=([0], [0])).reshape([1, 3, 2])
#     rightVec = np.array([0, 1])
#     rightTensor = np.tensordot(baseTensor, rightVec, axes=([2], [0])).reshape([2, 3, 1])
#
#     psi = [tn.Node(leftTensor)] + [tn.Node(baseTensor) for i in range(N-2)] + [tn.Node(rightTensor)]
#     psi[-1].tensor /= np.sqrt(bops.getOverlap(psi, psi))
#     return psi


# Eq. 12 here - https://inspirehep.net/files/4c4f8bef45a20ca059100bea16a33fbb
def getAKLTState(N):
    # baseTensor = np.zeros((3, basicdefs.d, 3), dtype=complex)
    # baseTensor[0, 0, 1] = -1
    # baseTensor[1, 0, 0] = 1
    # baseTensor[0, 1, 2] = -1
    # baseTensor[2, 1, 0] = 1
    # baseTensor[1, 2, 2] = -1
    # baseTensor[2, 2, 1] = 1
    # baseTensor /= np.sqrt(2)
    baseTensor = np.zeros((2, 3, 2), dtype=complex)
    baseTensor[1, 0, 0] = -np.sqrt(2/3)
    baseTensor[0, 1, 0] = -np.sqrt(1/3)
    baseTensor[1, 1, 1] = np.sqrt(1/3)
    baseTensor[0, 2, 1] = np.sqrt(2/3)
    leftVec = np.array([1, 1])
    leftTensor = np.tensordot(leftVec, baseTensor, axes=([0], [0])).reshape([1, 3, 2])
    rightVec = np.array([1, 0])
    rightTensor = np.tensordot(baseTensor, rightVec, axes=([2], [0])).reshape([2, 3, 1])
    psi = [tn.Node(leftTensor)] + [tn.Node(baseTensor) for i in range(N-2)] + [tn.Node(rightTensor)]
    psi[-1].tensor /= np.sqrt(bops.getOverlap(psi, psi))
    return psi



def getLocalT(a1, a2):
    return basicdefs.omega ** (- a1 * a2 / 2) * \
           np.matmul(np.linalg.matrix_power(basicdefs.pauliZ, a1),
                     np.linalg.matrix_power(basicdefs.pauliX, a2))


def getLocalA0():
    mat = np.zeros((basicdefs.d, basicdefs.d), dtype=complex)
    for a1 in range(basicdefs.d):
        for a2 in range(basicdefs.d):
            curr = getLocalT(a1, a2)
            mat += curr
    mat /= basicdefs.d
    return tn.Node(mat)


def wignerFunction(uVec, psi):
    a1s = uVec[0]
    a2s = uVec[1]
    psiTu = bops.copyState(psi)
    psiTuCopy = bops.copyState(psiTu)
    A0 = tn.Node(getLocalA0())
    for i in range(len(psi)):
        Tu = tn.Node(getLocalT(a1s[i], a2s[i]))
        AU = bops.multiContraction(bops.multiContraction(Tu, A0, '1', '0'), Tu, '1', '1*')
        psiTuCopy[i] = bops.permute(
            bops.multiContraction(psiTuCopy[i], AU, '1', '1'), [0, 2, 1])
    return bops.getOverlap(psiTu, psiTuCopy)


def getBasicState(N):
    baseTensor = np.ones((1, basicdefs.d, 1), dtype=complex) / np.sqrt(basicdefs.d)
    psi = [tn.Node(baseTensor) for i in range(N)]
    psi[N-1].tensor /= np.sqrt(bops.getOverlap(psi, psi))
    return psi


def getTGate():
    mat = np.eye(3, dtype=complex)
    mat[1, 1] *= np.exp(1j * 2 * np.pi / (basicdefs.d * 4))
    mat[2, 2] *= np.exp(2 * 1j * 2 * np.pi / (basicdefs.d * 4))
    return tn.Node(mat)


def getMana(psi, us):
    manaSum = 0
    for j in range(len(us)):
        u = us[j]
        wu = wignerFunction(u, psi)
        manaSum += np.abs(wu) / basicdefs.d ** len(psi)
    return np.log(manaSum)


def getUs(N):
    us = [np.zeros(2*N, dtype=int) for j in range(basicdefs.d**(2*N))]
    for j in range(basicdefs.d**(2*N)):
        trinary = int2base(j, length=2*N)
        for i in range(2*N):
            us[j][i] = int(trinary[i])
        us[j] = us[j].reshape([2, N])
    return us


def toVirtualHalfSpins(psi: List[tn.Node]):
    splitter = np.zeros((3, 4))
    splitter[0, 0] = 1
    splitter[1, 3] = 1
    splitter[2, 1] = 1 / np.sqrt(2)
    splitter[2, 2] = -1 / np.sqrt(2)
    splitter = tn.Node(splitter.reshape([3, 2, 2]))
    result = []
    for i in range(len(psi)):
        curr = bops.permute(bops.multiContraction(psi[i], splitter, '1', '0'), [0, 2, 3, 1])
        [r, l, te] = bops.svdTruncation(curr, [0, 1], [2, 3], '>>')
        result.append(r)
        result.append(l)
    return result


d = 3
n = 16
# with open('results/aklt_gs_D2', 'rb') as f:
#     gs2 = pickle.load(f)
# renyi.getSecondRenyi(gs2, d)
# renyi.getSecondRenyiFromRandomVecs(gs2, d)
Js = [1 / 3] # np.sort([-1 / i for i in range(1, 10)] + [0] + [1 / i for i in range(1, 10)]) * 1/15 + 1/3
Es = np.zeros(len(Js), dtype=complex)
p2s = np.zeros(len(Js), dtype=complex)
for i in range(len(Js)):
    J = Js[i]
    SPlus = np.zeros((d, d), dtype=complex)
    SPlus[1, 0] = 1
    SPlus[2, 1] = 1
    SMinus = SPlus.transpose()
    SX = (SPlus + SMinus) / np.sqrt(2)
    SY = 1j * (SPlus - SMinus) / np.sqrt(2)
    SZ = np.diag([-1, 0, 1])
    SDotS = np.kron(SX, SX) + np.kron(SY, SY) + np.kron(SZ, SZ)
    localTerm = SDotS + J * np.linalg.matrix_power(SDotS, 2)
    H = dmrg.getDMRGH(n, [np.zeros((d, d)) for i in range(n)], [np.copy(localTerm) for i in range(n - 1)], d=d)
    psi0 = bops.getStartupState(n, mode='aklt')
    psi0Copy = bops.copyState(psi0)
    HLs, HRs = dmrg.getHLRs(H, psi0Copy)
    gs128, E0, truncErrs = dmrg.getGroundState(H, HLs, HRs, psi0Copy, maxBondDim=128)
    psi0Copy = bops.copyState(psi0)
    HLs, HRs = dmrg.getHLRs(H, psi0Copy)
    gs2, E0, truncErrs = dmrg.getGroundState(H, HLs, HRs, psi0Copy, maxBondDim=2)
    if J == 1/3:
        print(renyi.getSecondRenyi(gs2, d))
        renyi.getSecondRenyiFromRandomVecs(gs2, d)
    Es[i] = dmrg.stateEnergy(gs128, H)
    p2s[i] = bops.getRenyiEntropy(gs128, 2, int(n/2))
    print([J, Es[i], p2s[i]])
plt.plot(Js, Es)
plt.plot(Js, p2s)
plt.show()
psi2, trPs = renyi.get2PsiVectorized(gs128, d=d)
