import pickle
import numpy as np
import basicOperations as bops
import tensornetwork as tn
import pepsExpect as pe
import matplotlib.pyplot as plt
import sys

# Linear regression, based on https://stackoverflow.com/questions/6148207/linear-regression-with-matplotlib-numpy
def linearRegression(Ns, Vs):
    coef = np.polyfit(Ns, np.log2(Vs), 1)
    print(coef)
    poly1d_fn = np.poly1d(coef)
    # plt.plot(Ns, Vs, 'yo', Ns, 2**poly1d_fn(Ns), '--k', color=color, label='p2')
    plt.scatter(Ns, Vs)
    plt.plot(Ns, 2 ** poly1d_fn(Ns), '--k')
    plt.yscale('log')
    plt.xticks(Ns)
    plt.show()
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

def toricVar(ls: np.array):
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


    hs = ls / 2
    vs = np.zeros(len(hs))
    v2s = np.zeros(len(hs))
    for i in range(len(hs)):
        h = hs[i]
        norm = pe.applyLocalOperators(cUp, dUp, cDown, dDown, leftRow, rightRow, A, B, w, h,
                                       [tn.Node(np.eye(d)) for i in range(w * h)])
        leftRow = bops.multNode(leftRow, 1 / norm**(2/w))

        doubleLeftRow = getDoubleRow(leftRow, chi, d)
        ops = [tn.Node(eMat) for i in range(w * h)]
        vs[i] = pe.applyLocalOperators(doubleCUp, doubleDUp, doubleCDown, doubleDDown, doubleLeftRow, doubleRightRow, doubleA, doubleB, w, h, ops)

        # qLeftRow = getDoubleRow(doubleLeftRow, chi**2, d**2)
        # ops = [tn.Node(np.eye(d**4)) for i in range(w * h)]
        # v2s[i] = pe.applyLocalOperators(qCUp, qDUp, qCDown, qDDown, qLeftRow, qRightRow, qA, qB, w, h, ops)

    linearRegression(ls, vs)

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
            for k in range(NA * 2 - 1, NA - 1, -1):
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

statesDir = sys.argv[1]
outDir = sys.argv[2]
NAs = [int(N) for N in sys.argv[3:]]
XXVar(statesDir, outDir, NAs)
