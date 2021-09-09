import numpy as np
import basicOperations as bops
import tensornetwork as tn

d = 3
omega = np.exp(1j * 2 * np.pi / d)
pauliX = np.zeros((d, d), dtype=complex)
pauliX[0, -1] = 1
pauliX[1, 0] = 1
pauliX[2, 1] = 1
pauliZ = np.diag(np.array([omega**j for j in range(d)]))


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


def getAKLTState(N):
    sigmaX = np.zeros((2, 2), dtype=complex)
    sigmaX[1, 0] = 1
    sigmaX[0, 1] = 1
    sigmaY = np.zeros((2, 2), dtype=complex)
    sigmaY[0, 1] = -1j
    sigmaY[1, 0] = 1j
    sigmaPlus = 0.5 * (sigmaX + sigmaY)
    sigmaMinus = 0.5 * (sigmaX - sigmaY)
    sigmaZ = np.eye(2, dtype=complex)
    sigmaZ[1, 1] = -1
    baseTensor = np.zeros((2, 3, 2), dtype=complex)
    baseTensor[:, 0, :] = sigmaPlus * np.sqrt(2)
    baseTensor[:, 1, :] = -sigmaMinus * np.sqrt(2)
    baseTensor[:, 2, :] = sigmaZ

    leftVec = np.array([1, 0])
    leftTensor = np.reshape(np.tensordot(leftVec, baseTensor, axes=([0], [0])), [1, 3, 2])
    rightVec = np.array([0, 1])
    rightTensor = np.reshape(np.tensordot(baseTensor, rightVec, axes=([2], [0])), [2, 3, 1])

    psi = [tn.Node(leftTensor)] + [tn.Node(baseTensor) for i in range(N-2)] + [tn.Node(rightTensor)]
    psi[-1].tensor /= np.sqrt(bops.getOverlap(psi, psi))
    return psi


def getLocalT(a1, a2):
    return omega ** (- a1 * a2 / 2) * \
           np.matmul(np.linalg.matrix_power(pauliZ, a1),
                     np.linalg.matrix_power(pauliX, a2))


def getLocalA0():
    mat = np.zeros((d, d), dtype=complex)
    for a1 in range(d):
        for a2 in range(d):
            curr = getLocalT(a1, a2)
            mat += curr
    mat /= d
    return tn.Node(mat)


def wignerFunction(uVec, psi):
    a1s = uVec[0]
    a2s = uVec[1]
    psiTu = bops.copyState(psi)
    for i in range(len(psi)):
        Tu = tn.Node(getLocalT(a1s[i], a2s[i]))
        psiTu[i] = bops.permute(
            bops.multiContraction(psiTu[i], Tu, '1', '0'), [0, 2, 1])
    psiTuCopy = bops.copyState(psiTu)
    A0 = tn.Node(getLocalA0())
    for i in range(len(psi)):
        psiTuCopy[i] = bops.permute(
            bops.multiContraction(psiTuCopy[i], A0, '1', '0'), [0, 2, 1])
    return bops.getOverlap(psiTu, psiTuCopy)


def getBasicState(N):
    baseTensor = np.ones((1, d, 1), dtype=complex)
    psi = [tn.Node(baseTensor) for i in range(N)]
    psi[-1] /= np.sqrt(bops.getOverlap(psi, psi))
    return psi


def getTGate():
    mat = np.eye(3, dtype=complex)
    mat[1, 1] *= np.exp(1j * 2 * np.pi / (d * 2))
    mat[2, 2] *= np.exp(2 * 1j * 2 * np.pi / (d * 2))
    return tn.Node(mat)

def getMana(psi, us):
    mana = -1
    for j in range(len(us)):
        u = us[j]
        wu = wignerFunction(u, psi)
        mana += np.abs(wu) / d ** N
    return mana

N = 8
psi = getAKLTState(N)
# u = [[0, 0], [0, 0]]
# wu = wignerFunction(u, psi)
us = [np.zeros(2*N, dtype=int) for j in range(d**(2*N))]
for j in range(d**(2*N)):
    trinary = int2base(j, length=2*N)
    for i in range(2*N):
        us[j][i] = int(trinary[i])
    us[j] = us[j].reshape([2, N])
print(getMana(psi, us))
T = getTGate()
psi[2] = bops.permute(bops.multiContraction(psi[2], T, '1', '0'), [0, 2, 1])
print(getMana(psi, us))
psi[2] = bops.permute(bops.multiContraction(psi[2], T, '1', '0'), [0, 2, 1])
print(getMana(psi, us))
b = 1