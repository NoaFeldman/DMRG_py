import numpy as np
import tensornetwork as tn
from typing import List
import matplotlib.pyplot as plt
import pickle
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/noa/PycharmProjects/DMRG_py')
import basicOperations as bops
import DMRG as dmrg
import magic.magicRenyi as renyi
import magic.basicDefs as basicdefs
import torch


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


# Eq. 12 here - https://inspirehep.net/files/4c4f8bef45a20ca059100bea16a33fbb
def getAKLTState(N):
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
n = 8
stepSize = 0.05
jRange = 2 / 3
Js = [np.round(stepSize * i, 3) for i in range(int(jRange / stepSize))]
Es = np.zeros(len(Js), dtype=complex)
p2s = np.zeros(len(Js), dtype=complex)
m2s = np.zeros(len(Js), dtype=complex)
SPlus = np.zeros((d, d), dtype=complex)
SPlus[1, 0] = 1
SPlus[2, 1] = 1
SMinus = SPlus.transpose()
SX = (SPlus + SMinus) / np.sqrt(2)
SY = 1j * (SPlus - SMinus) / np.sqrt(2)
SZ = np.diag([-1, 0, 1])
SDotS = np.kron(SX, SX) + np.kron(SY, SY) + np.kron(SZ, SZ)
psi0 = bops.getStartupState(n, mode='aklt')
for i in range(len(Js)):
   J = Js[i]
   localTerm = SDotS + J * np.linalg.matrix_power(SDotS, 2)
   gs, E0, truncErrs = dmrg.DMRG(psi0, [np.zeros((d, d), dtype=complex) for i in range(n)],
                                        [np.copy(localTerm) for i in range(n - 1)], d=d)
   psi0 = gs
   Es[i] = E0
   if J == 0.3:
       b = 1
   gs16 = bops.relaxState(gs, 16)
   gs8 = bops.relaxState(gs, 8)
   print([bops.getOverlap(gs, gs16), bops.getOverlap(gs, gs8)])
   p2s[i] = bops.getRenyiEntropy(gs, 2, int(n/2))
   print(J, gs[int(n/2)][2].dimension)
   if gs[int(n/2)][2].dimension > 16:
       speedup = False
   else:
       speedup = True
   # renyi.getSecondRenyiFromRandomVecs(gs, d, outdir='results/haldane_J_' + str(np.round(J, 5)), rep=1, speedup=speedup)
plt.plot(Js, p2s)
plt.scatter(Js, p2s)
plt.show()


def torchFromNumpy(psi):
    res = []
    for i in range(len(psi)):
        res.append(tn.Node(torch.from_numpy(psi[i].tensor)))
    return res

J = 1/3
n = 4
psi0 = bops.getStartupState(n, mode='aklt')
localTerm = SDotS + J * np.linalg.matrix_power(SDotS, 2)
gs, E0, truncErrs = dmrg.DMRG(psi0, [np.zeros((d, d), dtype=complex) for i in range(n)],
                              [np.copy(localTerm) for i in range(n - 1)], d=d)
renyi.getSecondRenyi(gs, d)
# bops.init('pytorch', 'cpu')
# gs = torchFromNumpy(gs)
renyi.getSecondRenyiFromRandomVecs(gs, d=d, outdir='renyi2_speedup_' + str(n), rep=int(sys.argv[1]), speedup=True)
