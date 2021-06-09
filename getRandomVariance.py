import pickle
import numpy as np
import basicOperations as bops
import tensornetwork as tn
import pepsExpect as pe
import matplotlib.pyplot as plt
import sys
import basicAnalysis as ban

eMat = np.eye(4)
eMat[1, 2] = 1
eMat[2, 1] = 1
E = tn.Node(eMat)
X = np.zeros((2, 2))
X[0, 1] = 1
X[1, 0] = 1
Z = np.zeros((2, 2))
Z[0, 0] = 1
Z[1, 1] = -1
Y = np.zeros((2, 2), dtype=complex)
Y[0, 1] = -1j
Y[1, 0] = 1j
swap = np.zeros((4, 4))
swap[0, 0] = 1
swap[1, 2] = 1
swap[2, 1] = 1
swap[3, 3] = 1
T = np.reshape([1, 1, 0, 2 ** (-4), 2 ** (-4), 2 ** (-4), 2 ** (-4), 2 ** (-4), 2 ** (-4)], [3, 3])
fullT = np.zeros((9, 9))
fullT[0, 0] = 1
fullT[0, 4] = 1
fullT[4, 0] = 2**(-4)
fullT[4, 4] = 2**(-4)
fullT[4, 8] = 2**(-4)
fullT[8, 0] = 2**(-4)
fullT[8, 4] = 2**(-4)
fullT[8, 8] = 2**(-4)

digs = '012'
def int2base(x, base, N=None):
    if x == 0:
        res = '0'
    digits = []
    while x:
        digits.append(digs[int(x % base)])
        x = int(x / base)
    digits.reverse()
    res = ''.join(digits)
    if N is None:
        return res
    return '0' * (N - len(res)) + res


def expected(l):
    inVec = np.array([1, 1, 0])
    outVec = np.reshape(inVec, [1, 3])
    return np.matmul(outVec, np.matmul(np.linalg.matrix_power(T, l), inVec))[0] / 2

def expectedN(l, n):
    Tn = np.eye(1)
    for i in range(n):
        Tn = np.kron(Tn, T)
    inVec = np.ones(3**n)
    for j in range(len(inVec)):
        if str.find(int2base(j, 3), '2') > -1:
            inVec[j] = 0
    outVec = np.reshape(inVec, [1, len(inVec)])
    return np.matmul(outVec, np.matmul(np.linalg.matrix_power(Tn, l), inVec))[0] / 2**n


def toricVar(ls: np.array, op=E, color='blue'):
    w = 2
    d = 2
    chi = 4
    with open('results/toricBoundaries', 'rb') as f:
        [upRow, downRow, leftRow, rightRow, openA, openB, A, B] = pickle.load(f)
    [cUp, dUp, te] = bops.svdTruncation(upRow, [0, 1], [2, 3], '>>')
    [cDown, dDown, te] = bops.svdTruncation(downRow, [0, 1], [2, 3], '>>')

    def getDoubleOp(op, d):
        return tn.Node(np.reshape(np.transpose(np.reshape(np.outer(op.tensor, op.tensor), [d] * 10), [0, 5, 1, 6, 2, 7, 3, 8, 4, 9]),
                       [d**2] * 5))
    doubleA = getDoubleOp(A, d)
    doubleB = getDoubleOp(B, d)
    def getDoubleRow(origRow, chi, d):
        openRow = np.reshape(origRow.tensor, [chi, d, d, d, d, chi])
        doubleRow = tn.Node(
            np.reshape(np.transpose(np.reshape(np.outer(openRow, openRow), [chi, d, d, d, d, chi] + [chi, d, d, d, d, chi]),
                                    [0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11]),
                       [chi**2, d**4, d**4, chi**2]))
        return doubleRow

    doubleUpRow = getDoubleRow(upRow, chi, d)
    doubleDownRow = getDoubleRow(downRow, chi, d)
    doubleRightRow = getDoubleRow(rightRow, chi, d)
    [doubleCUp, doubleDUp, te] = bops.svdTruncation(doubleUpRow, [0, 1], [2, 3], '>>')
    [doubleCDown, doubleDDown, te] = bops.svdTruncation(doubleDownRow, [0, 1], [2, 3], '>>')

    # qA = getDoubleOp(doubleA, d ** 2)
    # qB = getDoubleOp(doubleB, d ** 2)
    # qUpRow = getDoubleRow(doubleUpRow, chi**2, d**2)
    # qDownRow = getDoubleRow(doubleDownRow, chi**2, d**2)
    # qRightRow = getDoubleRow(doubleRightRow, chi**2, d**2)
    # [qCUp, qDUp, te] = bops.svdTruncation(qUpRow, [0, 1], [2, 3], '>>')
    # [qCDown, qDDown, te] = bops.svdTruncation(qDownRow, [0, 1], [2, 3], '>>')


    hs = ls * 2
    vs = np.zeros(len(hs))
    v2s = np.zeros(len(hs))
    for i in range(len(hs)):
        h = hs[i]
        norm = pe.applyLocalOperators(cUp, dUp, cDown, dDown, leftRow, rightRow, A, B, w, h,
                                       [tn.Node(np.eye(d)) for i in range(w * h)])
        leftRow = bops.multNode(leftRow, 1 / norm**(2/w))

        doubleLeftRow = getDoubleRow(leftRow, chi, d)
        ops = [op for i in range(w * h)]
        vs[i] = pe.applyLocalOperators(doubleCUp, doubleDUp, doubleCDown, doubleDDown, doubleLeftRow, doubleRightRow, doubleA, doubleB, w, h, ops)

        # qLeftRow = getDoubleRow(doubleLeftRow, chi**2, d**2)
        # ops = [tn.Node(np.eye(d**4)) for i in range(w * h)]
        # v2s[i] = pe.applyLocalOperators(qCUp, qDUp, qCDown, qDDown, qLeftRow, qRightRow, qA, qB, w, h, ops)
    print(vs)
    print([expected(l) for l in ls])

    ban.linearRegression(ls * 4, vs, show=False, color=color)

def doubleMPSSite(site):
    return tn.Node(np.reshape(np.transpose(np.reshape(np.outer(site.tensor, site.tensor),
                            np.shape(site.tensor) + np.shape(site.tensor)), [0, 3, 1, 4, 2, 5]),
               [np.shape(site.tensor)[0]**2, np.shape(site.tensor)[1]**2, np.shape(site.tensor)[2]**2]))

def XXVar(statesDir: str, outDir: str, NAs):
    vs = np.zeros(len(NAs) * 2, dtype=complex)
    for i in range(len(NAs)):
        NA = NAs[i]
        with open(statesDir + 'psiXX_NA_' + str(NA) + '_NB_' + str(NA), 'rb') as f:
            psi = pickle.load(f)
            psiCopy = bops.copyState(psi)
            for k in range(NA * 2 - 1, 1, -1):
                psi = bops.shiftWorkingSite(psi, k, '<<', maxBondDim=64)
            vs[2 * i] = bops.getOverlap(psi, psiCopy)
            bops.removeState(psiCopy)
            bops.removeState(psi[NA:])
            psi = psi[:NA]
        doublePsi = [None for j in range(len(psi))]
        for j in range(len(psi)):
            doublePsi[j] = doubleMPSSite(psi[j])
        doubleCopy = bops.copyState(doublePsi)
        for j in range(NA):
            doublePsi[j] = bops.permute(bops.multiContraction(doublePsi[j], E, '1', '0', cleanOr1=True), [0, 2, 1])
        vs[2 * i + 1] = bops.getOverlap(doublePsi, doubleCopy)

    with open(outDir + 'XX_n1_Error', 'wb') as f:
        pickle.dump(vs, f)
    print(vs)

def printExpected():
    print([expectedN(1, n) for n in range(1, 5)])


def commuteWithXXXX(rep):
    return str.count(rep, '2') % 2 == 0

def commutesWithZZZZ(rep):
    return (str.count(rep, '1') + str.count(rep, '2')) % 2 == 0

def toOps(rep):
    Xs = np.zeros(2 * 2)
    Zs = np.zeros(2 * 2)
    for i in range(len(rep)):
        if rep[i] == '1':
            Xs[i] = 1
        elif rep[i] == '2':
            Xs[i] = 1
            Zs[i] = 1
    return Xs, Zs

def starsOK(repUp, repDown):
    if repUp[0] != repUp[1] or repDown[0] != repDown[1] or repUp[2] != repUp[3] or repDown[2] != repDown[3]:
        return False
    afterCommute = [int(repUp[i] != repDown[i]) for i in range(len(repUp))]
    if afterCommute != [0, 0, 0, 0] and afterCommute != [1, 1, 1, 1]:
        return False
    return True

def plaqsOK(repUp, repDown, plaqUpXs, plaqDownXs):
    if list(repUp).count(1) % 2 != 0:
        return False
    afterCommute = [int(repUp[i] != repDown[i]) for i in range(len(repUp))]
    if plaqUpXs[0] != plaqUpXs[1] or plaqDownXs[0] != plaqDownXs[1] or plaqUpXs[2] != plaqUpXs[3] or plaqDownXs[2] != plaqDownXs[3]:
        return False
    if afterCommute != [0, 0, 0, 0] and afterCommute != [1, 1, 1, 1]:
        return False
    return True

def toricTMatrix2():
    n = 2
    Tn = np.zeros((3**4, 3**4))
    mid = [int2base(x, 3, N=2) for x in range(3**2)]
    startVec = np.zeros(len(Tn))
    for i in range(len(Tn)):
        irep = int2base(i, 3, N=2*n)
        iUp = irep[:2]
        iDown = irep[2:]
        for j in range(len(Tn)):
            jrep = int2base(j, 3, N=2 * n)
            jUp = jrep[:2]
            jDown = jrep[2:]
            for mUp in mid:
                starUp = iUp + mUp
                plaqUp = mUp + jUp
                for mDown in mid:
                    starDown = iDown + mDown
                    plaqDown = mDown + jDown
                    starUpXs, starUpZs = toOps(starUp)
                    starDownXs, starDownZs = toOps(starDown)
                    plaqUpXs, plaqUpZs = toOps(plaqUp)
                    plaqDownXs, plaqDownZs = toOps(plaqDown)
                    fullUp = iUp + mUp + jUp
                    fullDown = iDown + mDown + jDown
                    # if fullUp[2:] in ['100000', '010000', '200000', '020000'] and fullDown[2:] in ['110000', '110000', '220000', '220000']: # 110010
                    #     Tn[j, i] += 1
                    if starsOK(starUpXs, starDownXs):
                        if plaqsOK(plaqUpZs, plaqDownZs, plaqUpXs, plaqDownXs):
                            curr = 2**(- str.count(plaqUp, '1') - str.count(plaqDown, '1')) * \
                                   (-2)**(- str.count(plaqUp, '2') - str.count(plaqDown, '2'))
                            Tn[j, i] += curr
                            # if fullUp[2:] == '0000' and fullDown[2:] == '0000':  # 110010
                            #     print(fullUp, fullDown, curr)
                            b = 1
        if irep in ['0000', '0011', '1100', '1111']:
            startVec[i] = 1
    startVec /= np.sqrt(np.matmul(startVec, startVec))
    return Tn, startVec

def toricMatrix3():
    n = 3
    Tn = np.kron(np.kron(fullT, fullT), fullT)
    mid = [int2base(x, 3, N=2) for x in range(3 ** 2)]
    for i in range(len(Tn)):
        irep = int2base(i, 3, N=2*n)
        iUp = irep[:2]
        iLev = irep[2:4]
        iDown = irep[4:]
        for j in range(len(Tn)):
            jrep = int2base(j, 3, N=2*n)
            jUp = jrep[:2]
            jLev = jrep[2:4]
            jDown = jrep[4:]
            for mUp in mid:
                starUp = iUp + mUp
                plaqUp = mUp + jUp
                for mLev in mid:
                    starLev = iLev + mLev
                    plaqLev = mLev + jLev
                    for mDown in mid:
                        starDown = iDown + mDown
                        plaqDown = mDown + jDown



T2, startVec = toricTMatrix2()
# T2 = np.round(T2, 2)
[w, v] = np.linalg.eig(T2)
print(max(w)**(1/4))
[w, v] = np.linalg.eig(np.transpose(T2))
print(max(w)**(1/4))
print((np.matmul(np.transpose(startVec), np.matmul(np.linalg.matrix_power(T2, 5), startVec)) / \
      np.matmul(np.transpose(startVec), np.matmul(np.linalg.matrix_power(T2, 4), startVec)))**(1/4))

# mat = np.eye(4) + 0.5*np.kron(X, X) + 0.5*np.kron(Y, Y)
# toricVar(np.array(range(1, 20)), tn.Node(mat))
# mat = np.eye(4) + 0.5*np.kron(X, X) + 0.5*np.kron(Z, Z)
# toricVar(np.array(range(1, 20)), tn.Node(mat), color='orange')
# mat = np.eye(4) + 0.5*np.kron(Y, Y) + 0.5*np.kron(Z, Z)
# toricVar(np.array(range(1, 20)), tn.Node(mat), color='green')
# plt.show()

# statesDir = sys.argv[1]
# outDir = sys.argv[2]
# NAs = [int(N) for N in sys.argv[3:]]
# XXVar(statesDir, outDir, NAs)
