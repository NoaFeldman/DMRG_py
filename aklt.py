import numpy as np
import tensornetwork as tn
from typing import List
import pickle
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/')
import basicOperations as bops
import magicRenyi as renyi

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
    return basicDefs.omega ** (- a1 * a2 / 2) * \
           np.matmul(np.linalg.matrix_power(basicDefs.pauliZ, a1),
                     np.linalg.matrix_power(basicDefs.pauliX, a2))


def getLocalA0():
    mat = np.zeros((basicDefs.d, basicDefs.d), dtype=complex)
    for a1 in range(basicDefs.d):
        for a2 in range(basicDefs.d):
            curr = getLocalT(a1, a2)
            mat += curr
    mat /= basicDefs.d
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
    baseTensor = np.ones((1, basicDefs.d, 1), dtype=complex) / np.sqrt(basicDefs.d)
    psi = [tn.Node(baseTensor) for i in range(N)]
    psi[N-1].tensor /= np.sqrt(bops.getOverlap(psi, psi))
    return psi


def getTGate():
    mat = np.eye(3, dtype=complex)
    mat[1, 1] *= np.exp(1j * 2 * np.pi / (basicDefs.d * 4))
    mat[2, 2] *= np.exp(2 * 1j * 2 * np.pi / (basicDefs.d * 4))
    return tn.Node(mat)


def getMana(psi, us):
    manaSum = 0
    for j in range(len(us)):
        u = us[j]
        wu = wignerFunction(u, psi)
        manaSum += np.abs(wu) / basicDefs.d ** len(psi)
    return np.log(manaSum)


def getUs(N):
    us = [np.zeros(2*N, dtype=int) for j in range(basicDefs.d ** (2 * N))]
    for j in range(basicDefs.d ** (2 * N)):
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
SPlus = np.zeros((d, d), dtype=complex)
SPlus[1, 0] = 1
SPlus[2, 1] = 1
SMinus = SPlus.transpose()
SX = (SPlus + SMinus) / np.sqrt(2)
SY = 1j * (SPlus - SMinus) / np.sqrt(2)
SZ = np.diag([-1, 0, 1])
SDotS = np.kron(SX, SX) + np.kron(SY, SY) + np.kron(SZ, SZ)
psi0 = bops.getStartupState(n, mode='aklt')

# Js = [np.round(stepSize * i, 3) for i in range(int(jRange / stepSize))]
rep = sys.argv[1]
J = np.round(stepSize * int(sys.argv[2]), 3)
indir = sys.argv[3]
outdir = sys.argv[4]
# localTerm = SDotS + J * np.linalg.matrix_power(SDotS, 2)
# gs, E0, truncErrs = dmrg.DMRG(psi0, [np.zeros((d, d), dtype=complex) for i in range(n)],
#                                      [np.copy(localTerm) for i in range(n - 1)], d=d)
with open(indir + '/psi_haldane_J_' + str(np.round(J, 5)) + '_15', 'rb') as f:
    gs = pickle.load(f)
renyi.getSecondRenyiFromRandomVecs(gs, d, outdir=outdir + '/haldane_J_' + str(np.round(J, 5)),
                                   rep=rep, speedup=True)
test = False
if test:
    n = 4
    psi0 = bops.getStartupState(n, d=3, mode='aklt')
    op = np.eye(3, dtype=complex)
    op[1, 1] *= basicDefs.omega ** (1 / 4)
    op[1, 1] *= basicDefs.omega ** (2 / 4)
    bops.applySingleSiteOp(psi0, tn.Node(op), 1)
    renyi.getSecondRenyi(psi0, d)
    renyi.getSecondRenyiExact(psi0, d)
    renyi.getSecondRenyiFromRandomVecs(psi0, d, outdir='results/renyi2_speedup_4', speedup=True)
