from typing import Optional

from scipy import linalg
import numpy as np

import basicOperations as bops
import randomMeasurements as rm
import sys
import tensornetwork as tn

chi = 32


def bmpsRowStep(gammaL, lambdaMid, gammaR, lambdaSide, envOp):
    row = bops.multiContraction(bops.multiContraction(
        bops.multiContraction(bops.multiContraction(lambdaSide, gammaL, '1', '0', isDiag1=True),
                              lambdaMid, '2', '0', cleanOr1=True, cleanOr2=True, isDiag2=True),
                              gammaR, '2', '0', cleanOr1=True, cleanOr2=True),
                              lambdaSide, '3', '0', cleanOr1=True, isDiag2=True)
    opRow = bops.permute(bops.multiContraction(row, envOp, '12', '01', cleanOr1=True), [0, 2, 4, 5, 1, 3])
    [U, S, V, truncErr] = bops.svdTruncation(opRow, [0, 1, 2], [3, 4, 5], dir='>*<', maxBondDim=chi)
    newLambdaMid = bops.multNode(S, 1 / np.sqrt(sum(S.tensor**2)))
    lambdaSideInv = tn.Node(np.array([1 / val if val > 1e-15 else 0 for val in lambdaSide.tensor], dtype=complex))
    newGammaL = bops.multiContraction(lambdaSideInv, U, '1', '0', cleanOr2=True, isDiag1=True)
    splitter = tn.Node(bops.getLegsSplitterTensor(newGammaL[0].dimension, newGammaL[1].dimension))
    newGammaL = bops.unifyLegs(newGammaL, 0, 1)
    newGammaR = bops.multiContraction(V, lambdaSideInv, '2', '0', cleanOr1=True, cleanOr2=True, isDiag2=True)
    newGammaR = bops.unifyLegs(newGammaR, 2, 3)
    newLambdaSide = bops.multiContraction(bops.multiContraction(
        lambdaSide, splitter, '1', '0', cleanOr1=True, isDiag1=True),
        splitter, '01', '01', cleanOr1=True, cleanOr2=True)
    temp = newLambdaSide
    newLambdaSide = tn.Node(np.diag(newLambdaSide.tensor))
    tn.remove_node(temp)
    return newGammaL, newLambdaMid, newGammaR, newLambdaSide

def fidelity(rho, sigma):
    if np.all(np.round(rho, 13) == np.round(sigma, 13)):
        return 1
    vals, u = np.linalg.eigh(rho)
    vals = vals.astype('complex')
    vals = np.sqrt(vals)
    uSigmaU = np.matmul(np.conj(np.transpose(u)), np.matmul(sigma, u))
    sqrtRhoSig = np.array([uSigmaU[i] * vals[i] for i in range(len(vals))])
    toTrace = np.transpose(np.array([sqrtRhoSig[:, i] * vals[i] for i in range(len(vals))]))
    vals = np.linalg.eigvalsh(toTrace)
    return sum(np.sqrt(vals))**2


def checkConvergence(oldGammaC, oldLambdaC, oldGammaD, oldLambdaD, GammaC, LambdaC, GammaD, LambdaD, d):
    dmC = np.round(getRowDM(GammaC, LambdaC, GammaD, LambdaD, 0, d), 16)
    oldDmC = np.round(getRowDM(oldGammaC, oldLambdaC, oldGammaD, oldLambdaD, 0, d), 16)
    dmD = np.round(getRowDM(GammaD, LambdaD, GammaC, LambdaC, 0, d), 16)
    oldDmD = np.round(getRowDM(oldGammaD, oldLambdaD, oldGammaC, oldLambdaC, 0, d), 16)
    return fidelity(dmC, oldDmC), fidelity(dmD, oldDmD)


def getRowDM(GammaL, LambdaL, GammaR, LambdaR, sites, d):
    c = bops.multiContraction(bops.multiContraction(LambdaR, GammaL, '1', '0', isDiag1=True),
                              LambdaL, '2', '0', isDiag2=True)
    row = bops.multiContraction(bops.multiContraction(c, GammaR, '2', '0'), LambdaR, '3', '0', isDiag2=True)
    for i in range(sites):
        row = bops.multiContraction(row, GammaL, [len(row.edges) - 1], [0])
        row = bops.multiContraction(row, LambdaL, [len(row.edges) - 1], [0], isDiag2=True)
        row = bops.multiContraction(row, GammaR, [len(row.edges) - 1], [0])
        row = bops.multiContraction(row, LambdaR, [len(row.edges) - 1], [0], isDiag2=True)
    dm = bops.multiContraction(row, row, [0, len(row.edges) - 1], [0, len(row.edges) - 1, '*'])
    rho = np.reshape(dm.tensor, [d**((2 + 2 * sites)), d**((2 + 2 * sites))])
    return rho / np.trace(rho)


def getBMPSRowOps(GammaC, LambdaC, GammaD, LambdaD, AEnv, BEnv, steps):
    convergence = []
    envOpAB = bops.permute(bops.multiContraction(AEnv, BEnv, '1', '3'), [0, 3, 2, 4, 1, 5])
    envOpBA = bops.permute(bops.multiContraction(BEnv, AEnv, '1', '3'), [0, 3, 2, 4, 1, 5])
    op = envOpAB
    for i in range(steps):
        oldGammaC, oldLambdaC, oldGammaD, oldLambdaD = GammaC, LambdaC, GammaD, LambdaD
        GammaC, LambdaC, GammaD, LambdaD = bmpsRowStep(GammaC, LambdaC, GammaD, LambdaD, op)
        GammaD, LambdaD, GammaC, LambdaC = bmpsRowStep(GammaD, LambdaD, GammaC, LambdaC, op)
        # if i > 0:
        #     convergence.append(
        #         checkConvergence(oldGammaC, oldLambdaC, oldGammaD, oldLambdaD, GammaC, LambdaC, GammaD, LambdaD, 2))
        bops.removeState([oldGammaC, oldLambdaC, oldGammaD, oldLambdaD])

    cUp = bops.multiContraction(GammaC, LambdaC, '2', '0', isDiag2=True)
    dUp = bops.multiContraction(GammaD, LambdaD, '2', '0', isDiag2=True)
    GammaC, LambdaC, GammaD, LambdaD = bmpsRowStep(GammaC, LambdaC, GammaD, LambdaD, op)
    cDown = bops.multiContraction(GammaC, LambdaC, '2', '0', isDiag2=True)
    dDown = bops.multiContraction(GammaD, LambdaD, '2', '0', isDiag2=True)
    bops.removeState([GammaC, LambdaC, GammaD, LambdaD, oldGammaC, oldLambdaC, oldGammaD, oldLambdaD])
    return cUp, dUp, cDown, dDown


def bmpsSides(cUp: tn.Node, dUp: tn.Node, cDown: tn.Node, dDown: tn.Node, AEnv: tn.Node, BEnv: tn.Node, steps,
                 option='right'):
    envOpAB = bops.permute(bops.multiContraction(AEnv, BEnv, '1', '3'), [0, 3, 2, 4, 1, 5])
    upRow = bops.multiContraction(cUp, dUp, '2', '0')
    downRow = bops.multiContraction(cDown, dDown, '2', '0')
    if option == 'right':
        X = tn.Node(np.ones((upRow[3].dimension, envOpAB[3].dimension, downRow[3].dimension),
                            dtype=complex))
    else:
        X = tn.Node(np.ones((upRow[0].dimension, envOpAB[2].dimension, downRow[0].dimension),
                            dtype=complex))
    for i in range(steps):
        if option == 'right':
            X = bops.multiContraction(bops.multiContraction(bops.multiContraction(
                X, upRow, '0', '3'), envOpAB, '340', '013', cleanOr1=True), downRow, '034', '312')
        else:
            X = bops.multiContraction(bops.multiContraction(bops.multiContraction(
                X, upRow, '0', '0'), envOpAB, '023', '201', cleanOr1=True), downRow, '034', '012', cleanOr1=True)
        norm = np.sqrt(bops.multiContraction(X, X, '012', '012*').tensor)
        X = bops.multNode(X, 1 / norm)
    return X


# Start with a 2*2 DM, increase later
def bmpsCols(cUp: tn.Node, dUp: tn.Node, cDown: tn.Node, dDown: tn.Node, AEnv: tn.Node, BEnv: tn.Node, steps,
                 option='right'):
    envOpAB = bops.permute(bops.multiContraction(AEnv, BEnv, '1', '3'), [0, 3, 2, 4, 1, 5])
    envOpBA = bops.permute(bops.multiContraction(BEnv, AEnv, '1', '3'), [0, 3, 2, 4, 1, 5])
    upRow = bops.multiContraction(cUp, dUp, '2', '0')
    downRow = bops.multiContraction(cDown, dDown, '2', '0')
    if option == 'right':
        X = tn.Node(np.ones((upRow[3].dimension, envOpAB[3].dimension, envOpBA[3].dimension, downRow[3].dimension),
                            dtype=complex))
    else:
        X = tn.Node(np.ones((downRow[0].dimension, envOpBA[2].dimension, envOpAB[2].dimension, upRow[0].dimension),
                            dtype=complex))
    for i in range(steps):
        if option == 'right':
            X = bops.multiContraction(upRow, X, '3', '0', cleanOr1=True, cleanOr2=True)
            X = bops.multiContraction(X, downRow, '5', '3', cleanOr1=True, cleanOr2=True)
            X = bops.multiContraction(X, envOpAB, '123', '013', cleanOr1=True)
            X = bops.multiContraction(X, envOpBA, '67134', '01345', cleanOr1=True)
            X = bops.permute(X, [0, 2, 3, 1])
        else:
            X = bops.multiContraction(downRow, X, '0', '0', cleanOr1=True, cleanOr2=True)
            X = bops.multiContraction(X, upRow, '5', '0', cleanOr1=True, cleanOr2=True)
            X = bops.multiContraction(X, envOpAB, '456', '201', cleanOr1=True)
            X = bops.multiContraction(X, envOpBA, '67301', '01245')
            X = bops.permute(X, [0, 3, 2, 1])
        norm = np.sqrt(bops.multiContraction(X, X, '0123', '0123*').tensor)
        X = bops.multNode(X, 1 / norm)
    return X


def bmpsDensityMatrix(cUp, dUp, cDown, dDown, AEnv, BEnv, A, B, steps):
    rightRow = bmpsCols(cUp, dUp, cDown, dDown, AEnv, BEnv, steps, 'right')
    leftRow = bmpsCols(cUp, dUp, cDown, dDown, AEnv, BEnv, steps, 'left')
    upRow = bops.multiContraction(cUp, dUp, '2', '0')
    downRow = bops.multiContraction(cDown, dDown, '2', '0')
    circle = bops.multiContraction(
        bops.multiContraction(bops.multiContraction(leftRow, upRow, '3', '0'), rightRow, '5', '0'), downRow, '07', '03')


    parityTensor = np.eye(4, dtype=complex)
    parityTensor[1, 1] = -1
    parityTensor[3, 3] = -1
    parity = tn.Node(parityTensor)
    if A[0].dimension == 4:
        parityA = tn.Node(np.trace(bops.multiContraction(parity, A, '1', '0').tensor, axis1=0, axis2=5))
        ABNet = bops.permute(
            bops.multiContraction(bops.multiContraction(parityA, parityA, '1', '3'), bops.multiContraction(parityA, parityA, '1', '3'), '15', '03',
                                  cleanOr1=True, cleanOr2=True),
            [5, 1, 0, 2, 3, 6, 4, 7])
        p2 = bops.multiContraction(circle, ABNet, '01234567', '01234567')
        b = 1

    ABNet = bops.permute(
        bops.multiContraction(bops.multiContraction(B, A, '2', '4'), bops.multiContraction(A, B, '2', '4'), '28', '16',
                              cleanOr1=True, cleanOr2=True),
        [2, 10, 9, 13, 14, 5, 1, 6, 0, 4, 8, 12, 3, 7, 11, 15])
    dm = bops.multiContraction(circle, ABNet, '23140567', '01234567', cleanOr1=True, cleanOr2=True)
    ordered = np.round(np.reshape(dm.tensor, [16,  16]), 13)
    ordered = ordered / np.trace(ordered)
    return dm


# https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.109.020505
# temp
def twoCopiesEntanglement(circle, A, B):
    doubleCircle = tn.Node(np.kron(circle.tensor, circle.tensor))
    doubleA = tn.Node(np.kron(A.tensor, A.tensor))
    doubleB = tn.Node(np.kron(B.tensor, B.tensor))

    AEnv = tn.Node(np.trace(A.get_tensor(), axis1=0, axis2=5))
    BEnv = tn.Node(np.trace(B.get_tensor(), axis1=0, axis2=5))
    ABNet = bops.permute(bops.multiContraction(bops.multiContraction(BEnv, AEnv, '1', '3'),
                                               bops.multiContraction(AEnv, BEnv, '1', '3'), '15', '03'),
                         [5, 1, 0, 2, 3, 6, 4, 7])
    n = bops.multiContraction(circle, ABNet, '01234567', '01234567').tensor

    p2 = bops.multiContraction(doubleCircle, ABNet, '01234567', '01234567').tensor
    return p2

