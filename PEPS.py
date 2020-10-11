from typing import Optional

from scipy import linalg
import numpy as np

import basicOperations as bops
import randomMeasurements as rm
import sys
import tensornetwork as tn

chi = 32

def getBaseTheta(gammaA, lambdaA, gammaB, envOp, option) -> tn.Node:
    if option == 'horizontal':
        return bops.permute(bops.multiContraction(
            bops.multiContraction(gammaA, bops.multiContraction(lambdaA, gammaB, '1', '0'), '2', '0'),
            envOp, '12', '01'), [0, 2, 4, 5, 1, 3])
    else:
        return gammaA
        # TODO

# https://arxiv.org/pdf/0711.3960.pdf
def getTheta(baseTheta, lambdaB, bmpsOption, thetaType) -> tn.Node:
    if bmpsOption == 'horizontal':
        if thetaType == 1:
            Theta = bops.permute(bops.multiContraction(baseTheta, lambdaB, '4', '0'), [0, 1, 2, 3, 5, 4])
        elif thetaType == 2:
            Theta = bops.multiContraction(lambdaB, baseTheta, '1', '0')
        else:
            Theta = baseTheta
        Theta = bops.unifyLegs(bops.unifyLegs(Theta, 0, 1, cleanOriginal=False), 3, 4)
        if 0 in Theta.tensor.shape:
            b = 1
        return Theta
    elif bmpsOption == 'diagonal':
        # TODO
        return baseTheta


def getTridiagonal(M, dir):
    dim = M.edges[0].dimension
    betas = []
    alphas = []
    base = []
    counter = 0
    accuracy = 1e-12
    beta = 100
    formBeta = 200
    while beta > accuracy and counter < 50 and beta < formBeta:
        if counter == 0:
            vTensor = np.eye(dim) / np.sqrt(dim)
            vTensor[0, -1] = 1 / np.sqrt(dim)
            vTensor[-1, 0] = 1 / np.sqrt(dim)
            v = tn.Node(vTensor, backend=None)
            norm = np.sqrt(bops.multiContraction(v, v, '01', '01*').tensor)
            v = bops.multNode(v, 1 / norm)
        else:
            v = bops.multNode(w, 1 / beta)
        base.append(bops.copyState([v])[0])
        if dir == '>>':
            Mv = bops.multiContraction(M, v, '13', '01')
        else:
            Mv = bops.multiContraction(M, v, '02', '01')
        alpha = bops.multiContraction(v, Mv, '01', '01*').tensor * 1
        alphas.append(alpha)
        alphaV = bops.multNode(v, alpha)

        tn.remove_node(v)
        if counter > 0:
            tn.remove_node(w)
            betaFormV = bops.multNode(base[-2], beta)
            w = Mv - alphaV - betaFormV
            tn.remove_node(betaFormV)
        else:
            w = Mv - alphaV
        formBeta = beta
        beta = np.sqrt(bops.multiContraction(w, w, '01', '01*').tensor)
        betas.append(beta)
        counter += 1
    # TODO clean up
    return alphas, betas, base


def isDiagonal(m, accuracy=12):
    return np.count_nonzero(np.round(m - np.diag(np.diagonal(m)), accuracy)) == 0


def lanczos(M, dir):
    alphas, betas, base = getTridiagonal(M, dir)
    if len(betas) > 1:
        val, vec = linalg.eigh_tridiagonal(d=np.real(alphas), e=np.real(betas[:len(betas) - 1]), select='i', select_range=[len(alphas) - 1, len(alphas) - 1])
    else:
        val = alphas[0]
        vec = [1]
    res = bops.multNode(base[0], vec[0])
    for i in range(1, len(vec)):
        curr = res + bops.multNode(base[i], vec[i])
        tn.remove_node(res)
        res = curr
    return res

from scipy.sparse.linalg import eigs as largest_eigs
def largestEigenvector(M, dir):
    if dir == '>>':
        dim = M[1].dimension
        transposeInds = [0, 2, 1, 3]
    else:
        dim = M[0].dimension
        transposeInds = [1, 3, 0, 2]
    m = np.reshape(np.transpose(M.tensor, transposeInds), [M[0].dimension * M[1].dimension, M[0].dimension * M[1].dimension])
    w, v = largest_eigs(m, k=1)
    vTensor = np.reshape(v, [dim, dim])
    vTensor = (vTensor + np.conj(np.transpose(vTensor))) / 2
    res = tn.Node(vTensor, backend=None)
    norm = np.sqrt(bops.multiContraction(res,res,'01', '01*').tensor)
    return bops.multNode(res, 1 / norm), w


def bmpsRowStep(GammaL, LambdaL, GammaR, LambdaR, envOp, option):
    oldGammaL, oldLambdaL, oldGammaR, oldLambdaR = GammaL, LambdaL, GammaR, LambdaR
    baseTheta = getBaseTheta(GammaL, LambdaL, GammaR, envOp, option)
    theta1 = getTheta(baseTheta, LambdaR, option, thetaType=1)
    Mx = bops.multiContraction(theta1, theta1, '12', '12*')
    xvals = [-1]
    while max(np.abs(xvals) - xvals) > 1e-12:
        mx = np.reshape(np.transpose(Mx.tensor, [0, 2, 1, 3]), [Mx[0].dimension**2, Mx[0].dimension**2])
        v = np.sort(np.round(np.linalg.eigvals(mx), 4))
        vR, wx = largestEigenvector(Mx, '>>')
        xvals, ux = np.linalg.eigh(vR.get_tensor())
        xvals = np.round(xvals, 15)
        if xvals[0] < 0:
            xvals = -xvals
    xvals_sqrt = np.sqrt(xvals + 0j)
    xTensor = np.matmul(ux, np.diag(xvals_sqrt))
    X = tn.Node(xTensor)
    xValsInverseTensor = np.diag([1 / val if val > 1e-12 else 0 for val in xvals_sqrt])
    # xValsInverseTensor = np.diag([1 / val for val in xvals_sqrt])
    xInverseTensor = np.matmul(xValsInverseTensor, np.conj(np.transpose(ux)))
    XInverse = tn.Node(xInverseTensor)
    theta2 = getTheta(baseTheta, LambdaR, option, thetaType=2)
    My = bops.multiContraction(theta2, theta2, '12', '12*')
    yvals = [-1]
    while max(np.abs(yvals) - yvals) > 1e-12:
        vL, wy = largestEigenvector(My, '<<')
        yvals, uy = np.linalg.eigh(vL.get_tensor())
        yvals = np.round(yvals, 15)
        if yvals[0] < 0:
            yvals = -yvals
    yvals_sqrt = np.sqrt(yvals + 0j)
    yTensor = np.matmul(uy, np.diag(yvals_sqrt))
    Yt = tn.Node(yTensor)
    yValsInverseTensor = np.diag([1 / val if val > 1e-12 else 0 for val in yvals_sqrt])
    yInverseTensor = np.matmul(yValsInverseTensor, np.conj(np.transpose(uy)))
    YtInverse = tn.Node(yInverseTensor)
    splitterRight = tn.Node(bops.getLegsSplitterTensor(baseTheta[0].dimension, baseTheta[1].dimension))
    splitterLeft = tn.Node(bops.getLegsSplitterTensor(baseTheta[4].dimension, baseTheta[5].dimension))
    LambdaR = bops.multiContraction(splitterLeft,
                                    bops.multiContraction(LambdaR, splitterRight, '1', '0',
                                                          cleanOriginal1=True, cleanOriginal2=True), '01', '01',
                                    cleanOriginal1=True, cleanOriginal2=True)
    Yt[0] ^ LambdaR[0]
    LambdaR[1] ^ X[0]
    newLambda = tn.contract_between(tn.contract_between(Yt, LambdaR), X)
    [U, LambdaR, V, truncErr] = bops.svdTruncation(newLambda, [newLambda[0]], [newLambda[1]], dir='>*<', maxBondDim=chi)
    LambdaR = bops.multNode(LambdaR, 1 / np.sqrt(np.trace(np.power(LambdaR.tensor, 2))))

    theta = getTheta(baseTheta, LambdaR, option, thetaType=0)
    bops.removeState([GammaL, GammaR])
    LambdaRLeft = bops.copyState([LambdaR])[0]
    LambdaRRight = bops.copyState([LambdaR])[0]
    LambdaRLeft[1] ^ V[0]
    V[1] ^ XInverse[0]
    XInverse[1] ^ theta[0]
    theta[3] ^ YtInverse[1]
    YtInverse[0] ^ U[0]
    U[1] ^ LambdaRRight[0]
    Sigma = tn.contract_between(tn.contract_between(tn.contract_between(tn.contract_between(tn.contract_between( \
                                tn.contract_between( \
                                LambdaRLeft, V), XInverse), \
                                theta), YtInverse), U), LambdaRRight)
    [P, LambdaL, Q, truncErr] = bops.svdTruncation(Sigma, Sigma[:2], Sigma[2:], dir='>*<', maxBondDim=chi)
    LambdaL = bops.multNode(LambdaL, 1 / np.sqrt(np.trace(np.power(LambdaL.tensor, 2))))
    lambdaInverseTensor = 1 / np.diag(LambdaRLeft.tensor)
    lambdaInverseTensor[np.isnan(lambdaInverseTensor)] = 0
    LambdaRLeft = tn.Node(np.diag(lambdaInverseTensor))
    LambdaRRight = tn.Node(np.diag(lambdaInverseTensor))
    LambdaRLeft[1] ^ P[0]
    GammaL = tn.contract_between(LambdaRLeft, P)
    Q[2] ^ LambdaRRight[0]
    GammaR = tn.contract_between(Q, LambdaRRight)

    checkCannonization(GammaL, LambdaL, GammaR, LambdaR)

    return GammaL, LambdaL, GammaR, LambdaR


# TODO diagonal matrices -> vectors


def myBmpsRowStep(gammaL, lambdaMid, gammaR, lambdaSide, envOp):
    row = bops.multiContraction(bops.multiContraction(bops.multiContraction(bops.multiContraction(
                lambdaSide, gammaL, '1', '0'),
                lambdaMid, '2', '0', cleanOriginal1=True, cleanOriginal2=True),
                gammaR, '2', '0', cleanOriginal1=True, cleanOriginal2=True),
                lambdaSide, '3', '0', cleanOriginal1=True)
    opRow = bops.permute(bops.multiContraction(row, envOp, '12', '01', cleanOriginal1=True), [0, 2, 4, 5, 1, 3])
    [U, S, V, truncErr] = bops.svdTruncation(opRow, [opRow[0], opRow[1], opRow[2]], [opRow[3], opRow[4], opRow[5]], dir='>*<', maxBondDim=chi)
    newLambdaMid = bops.multNode(S, 1 / np.sqrt(sum(np.diag(S.tensor)**2)))
    lambdaSideInv = tn.Node(np.diag([1 / val if val > 1e-15 else 0 for val in np.diag(lambdaSide.tensor)]))
    newGammaL = bops.multiContraction(lambdaSideInv, U, '1', '0', cleanOriginal2=True)
    splitter = tn.Node(bops.getLegsSplitterTensor(newGammaL[0].dimension, newGammaL[1].dimension))
    newGammaL = bops.unifyLegs(newGammaL, 0, 1)
    newGammaR = bops.multiContraction(V, lambdaSideInv, '2', '0', cleanOriginal1=True, cleanOriginal2=True)
    newGammaR = bops.unifyLegs(newGammaR, 2, 3)
    newLambdaSide = bops.multiContraction(bops.multiContraction(lambdaSide, splitter, '1', '0', cleanOriginal1=True),
                                          splitter, '01', '01', cleanOriginal1=True, cleanOriginal2=True)
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
    c = bops.multiContraction(bops.multiContraction(LambdaR, GammaL, '1', '0'), LambdaL, '2', '0')
    row = bops.multiContraction(bops.multiContraction(c, GammaR, '2', '0'), LambdaR, '3', '0')
    for i in range(sites):
        row = bops.multiContraction(row, GammaL, [len(row.edges) - 1], [0])
        row = bops.multiContraction(row, LambdaL, [len(row.edges) - 1], [0])
        row = bops.multiContraction(row, GammaR, [len(row.edges) - 1], [0])
        row = bops.multiContraction(row, LambdaR, [len(row.edges) - 1], [0])
    dm = bops.multiContraction(row, row, [0, len(row.edges) - 1], [0, len(row.edges) - 1, '*'])
    rho = np.reshape(dm.tensor, [d**(2 * (2 + 2 * sites)), d**(2 * (2 + 2 * sites))])
    return rho / np.trace(rho)


def getBMPSRowOps(GammaC, LambdaC, GammaD, LambdaD, AEnv, BEnv, steps, tilingOption='horizontal', sideOption='up'):
    convergence = []
    if tilingOption == 'horizontal':
        envOpAB = bops.permute(bops.multiContraction(AEnv, BEnv, '1', '3'), [0, 3, 2, 4, 1, 5])
        envOpBA = bops.permute(bops.multiContraction(BEnv, AEnv, '1', '3'), [0, 3, 2, 4, 1, 5])
        if sideOption == 'down':
            envOpAB = bops.permute(envOpAB, [4, 5, 2, 3, 0, 1])
            envOpBA = bops.permute(envOpBA, [4, 5, 2, 3, 0, 1])
        if sideOption == 'up':
            op = envOpAB
        else:
            op = envOpBA
        b = 1
    for i in range(steps):
        oldGammaC, oldLambdaC, oldGammaD, oldLambdaD = GammaC, LambdaC, GammaD, LambdaD
        if tilingOption == 'horizontal':
            GammaC, LambdaC, GammaD, LambdaD = myBmpsRowStep(GammaC, LambdaC, GammaD, LambdaD, op)
            GammaD, LambdaD, GammaC, LambdaC = myBmpsRowStep(GammaD, LambdaD, GammaC, LambdaC, op)
        elif tilingOption == 'diagonal':
            GammaC, LambdaC, GammaD, LambdaD = bmpsRowStep(GammaC, LambdaC, GammaD, LambdaD, AEnv, tilingOption)
            GammaD, LambdaD, GammaC, LambdaC = bmpsRowStep(GammaD, LambdaD, GammaC, LambdaC, BEnv, tilingOption)
        if i > 0:
            convergence.append(
                checkConvergence(oldGammaC, oldLambdaC, oldGammaD, oldLambdaD, GammaC, LambdaC, GammaD, LambdaD, 2))
        bops.removeState([oldGammaC, oldLambdaC, oldGammaD, oldLambdaD])

    # TODO compare density matrices as a convergence step. Find a common way to do this for iMPS.
    # TODO (for toric code, did it manually and it works)
    if tilingOption == 'diagonal':
        cUp = bops.multiContraction(GammaD, LambdaD, '3', '0')
        dUp = bops.multiContraction(GammaC, LambdaC, '3', '0')
        oldGammaC, oldLambdaC, oldGammaD, oldLambdaD = GammaC, LambdaC, GammaD, LambdaD
        GammaC, LambdaC, GammaD, LambdaD = bmpsRowStep(GammaC, LambdaC, GammaD, LambdaD, AEnv, tilingOption)
        cDown = bops.multiContraction(GammaD, LambdaD, '3', '0')
        dDown = bops.multiContraction(GammaC, LambdaC, '3', '0')
        bops.removeState([GammaC, LambdaC, GammaD, LambdaD, oldGammaC, oldLambdaC, oldGammaD, oldLambdaD])
        return cUp, cDown, dUp, dDown
    if tilingOption == 'horizontal':
        cUp = bops.multiContraction(GammaC, LambdaC, '2', '0')
        dUp = bops.multiContraction(GammaD, LambdaD, '2', '0')
        GammaC, LambdaC, GammaD, LambdaD = myBmpsRowStep(GammaC, LambdaC, GammaD, LambdaD, op)
        cDown = bops.multiContraction(GammaC, LambdaC, '2', '0')
        dDown = bops.multiContraction(GammaD, LambdaD, '2', '0')
        bops.removeState([GammaC, LambdaC, GammaD, LambdaD, oldGammaC, oldLambdaC, oldGammaD, oldLambdaD])
        return cUp, dUp, cDown, dDown

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
            X = bops.multiContraction(upRow, X, '3', '0', cleanOriginal1=True, cleanOriginal2=True)
            X = bops.multiContraction(X, downRow, '5', '3', cleanOriginal1=True, cleanOriginal2=True)
            X = bops.multiContraction(X, envOpAB, '123', '013', cleanOriginal1=True)
            X = bops.multiContraction(X, envOpBA, '67134', '01345', cleanOriginal1=True)
            X = bops.permute(X, [0, 2, 3, 1])
        else:
            X = bops.multiContraction(downRow, X, '0', '0', cleanOriginal1=True, cleanOriginal2=True)
            X = bops.multiContraction(X, upRow, '5', '0', cleanOriginal1=True, cleanOriginal2=True)
            X = bops.multiContraction(X, envOpAB, '456', '201', cleanOriginal1=True)
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
        bops.multiContraction(bops.multiContraction(leftRow, upRow, '3', '0'), rightRow, '5', '0'),
        downRow, '07', '03')
    ABNet = bops.permute(bops.multiContraction(bops.multiContraction(B, A, '2', '4'),
                                  bops.multiContraction(A, B, '2', '4'), '28', '16', cleanOriginal2=True, cleanOriginal1=True),
                         [2, 10, 9, 13, 14, 5, 1, 6, 0, 4, 8, 12, 3, 7, 11, 15])
    dm = bops.multiContraction(circle, ABNet, '23140567', '01234567', cleanOriginal1=True, cleanOriginal2=True)

    ordered = np.round(np.reshape(dm.tensor, [16,  16]), 13)
    ordered = ordered / np.trace(ordered)
    return dm

