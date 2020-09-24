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


def largestEigenvector(M, dir):
    from scipy.sparse.linalg import eigs as largest_eigs
    dim = M.edges[0].dimension
    if dir == '>>':
        transposeInds = [0, 2, 1, 3]
    else:
        transposeInds = [1, 3, 0, 2]
    m = np.reshape(np.transpose(M.tensor, transposeInds), [dim**2, dim**2])
    w, v = largest_eigs(m, k=1)
    vTensor = np.reshape(v, [dim, dim])
    vTensor = (vTensor + np.conj(np.transpose(vTensor))) / 2
    # # TODO just cheking
    # vTensor = np.real(vTensor)
    res = tn.Node(vTensor, backend=None)
    norm = np.sqrt(bops.multiContraction(res,res,'01', '01*').tensor)
    return bops.multNode(res, 1 / norm), w


def bmpsRowStep(GammaL, LambdaL, GammaR, LambdaR, envOp, option):
    oldGammaL, oldLambdaL, oldGammaR, oldLambdaR = GammaL, LambdaL, GammaR, LambdaR
    baseTheta = getBaseTheta(GammaL, LambdaL, GammaR, envOp, option)
    theta1 = getTheta(baseTheta, LambdaR, option, thetaType=1)
    Mx = bops.multiContraction(theta1, theta1, '12', '12*')
    xvals = [-1]
    while max(np.abs(xvals) - xvals) > 1e-12 or min(np.abs(xvals)) < 1e-8:
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
    yvals_sqrt = np.sqrt(yvals + 0j)
    yTensor = np.matmul(uy, np.diag(yvals_sqrt))
    Yt = tn.Node(yTensor)
    yValsInverseTensor = np.diag([1 / val for val in yvals_sqrt])
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


def checkCannonization(GammaC, LambdaC, GammaD, LambdaD):
    c = bops.multiContraction(GammaC, LambdaC, '2', '0')
    id = np.round(bops.multiContraction(c, c, '12', '12*').tensor, 3)
    if not (np.all(np.diag(np.diag(id)) == id) and set(np.diag(id)).issubset({0, 1})):
        return False
    c = bops.multiContraction(LambdaD, GammaC, '1', '0')
    id = np.round(bops.multiContraction(c, c, '01', '01*').tensor, 3)
    if not (np.all(np.diag(np.diag(id)) == id) and set(np.diag(id)).issubset({0, 1})):
        return False
    c = bops.multiContraction(GammaD, LambdaD, '2', '0')
    id = np.round(bops.multiContraction(c, c, '12', '12*').tensor, 3)
    if not (np.all(np.diag(np.diag(id)) == id) and set(np.diag(id)).issubset({0, 1})):
        return False
    c = bops.multiContraction(LambdaC, GammaD, '1', '0')
    id = np.round(bops.multiContraction(c, c, '01', '01*').tensor, 3)
    if not (np.all(np.diag(np.diag(id)) == id) and set(np.diag(id)).issubset({0, 1})):
        return False
    return True


def myBmpsRowStep(gammaL, lambdaMid, gammaR, lambdaSide, envOp):
    row = bops.multiContraction(bops.multiContraction(bops.multiContraction(bops.multiContraction(
                lambdaSide, gammaL, '1', '0'),
                lambdaMid, '2', '0', cleanOriginal1=True, cleanOriginal2=True),
                gammaR, '2', '0', cleanOriginal1=True, cleanOriginal2=True),
                lambdaSide, '3', '0', cleanOriginal1=True, cleanOriginal2=True)
    opRow = bops.permute(bops.multiContraction(row, envOp, '12', '01', cleanOriginal1=True), [0, 2, 4, 5, 1, 3])
    opRow = bops.unifyLegs(bops.unifyLegs(opRow, 4, 5), 0, 1)
    [U, lambdaSide, V, truncErr] = bops.svdTruncation(opRow, [opRow[0], opRow[1]], [opRow[2], opRow[3]], dir='>*<', maxBondDim=chi)
    lambdaSide = bops.multNode(lambdaSide, 1 / np.sqrt(sum(np.power(np.diag(lambdaSide.tensor), 2))))
    sV = bops.multiContraction(lambdaSide, V, '1', '0')
    temp = bops.multiContraction(sV, sV, '01', '01*')
    lambdaL = np.diag(np.sqrt(np.diag(temp.tensor + 0j)))
    lambdaLInverse = np.diag(np.array([1 / v for v in np.diag(lambdaL)]))
    gammaL = bops.multiContraction(V, tn.Node(lambdaLInverse), '2', '0', cleanOriginal1=True, cleanOriginal2=True)
    uS = bops.multiContraction(U, lambdaSide, '2', '0')
    temp2 = bops.multiContraction(uS, uS, '12', '12*')
    lambdaR = np.diag(np.sqrt(np.diag(temp2.tensor + 0j)))
    lambdaRInverse = np.diag(np.array([1 / v for v in np.diag(lambdaR)]))
    gammaR = bops.multiContraction(tn.Node(lambdaRInverse), U, '1', '0', cleanOriginal1=True, cleanOriginal2=True)
    lambdaMid = bops.multiContraction(tn.Node(lambdaR), tn.Node(lambdaL), '1', '0', cleanOriginal1=True, cleanOriginal2=True)
    lambdaMid = bops.multNode(lambdaMid, 1 / np.sqrt(sum(np.power(np.diag(lambdaMid.tensor), 2))))
    return gammaL, lambdaMid, gammaR, lambdaSide



def checkConvergence(oldGammaC, oldLambdaC, oldGammaD, oldLambdaD, GammaC, LambdaC, GammaD, LambdaD, d):
    dmC = getRowDM(GammaC, LambdaC, GammaD, LambdaD, 1, d)
    oldDmC = getRowDM(oldGammaC, oldLambdaC, oldGammaD, oldLambdaD, 1, d)
    dmD = getRowDM(GammaD, LambdaD, GammaC, LambdaC, 1, d)
    oldDmD = getRowDM(oldGammaD, oldLambdaD, oldGammaC, oldLambdaC, 1, d)
    return max(map(max, dmC - oldDmC)) / max(map(max, dmC)), max(map(max, dmD - oldDmD)) / max(map(max, dmD))


def getRowDM(GammaL, LambdaL, GammaR, LambdaR, sites, d):
    c = bops.multiContraction(bops.multiContraction(LambdaR, GammaL, '1', '0'), LambdaL, '2', '0')
    row = bops.multiContraction(bops.multiContraction(c, GammaR, '2', '0'), LambdaR, '3', '0')
    for i in range(sites):
        row = bops.multiContraction(row, GammaL, [len(row.edges) - 1], [0])
        row = bops.multiContraction(row, LambdaL, [len(row.edges) - 1], [0])
        row = bops.multiContraction(row, GammaR, [len(row.edges) - 1], [0])
        row = bops.multiContraction(row, LambdaR, [len(row.edges) - 1], [0])
    dm = bops.multiContraction(row, row, [0, len(row.edges) - 1], [0, len(row.edges) - 1, '*'])
    return np.reshape(dm.tensor, [d**(2 + 2 * sites), d**(2 + 2 * sites)])


def getBMPSRowOps(GammaC, LambdaC, GammaD, LambdaD, AEnv, BEnv, steps, option='horizontal'):
    convergence = []
    if option == 'horizontal':
        envOpAB = bops.permute(bops.multiContraction(AEnv, BEnv, '1', '3'), [0, 3, 2, 4, 1, 5])
        # envOpBA = bops.permute(bops.multiContraction(BEnv, AEnv, '1', '3'), [0, 3, 2, 4, 1, 5])
    for i in range(2 * steps):
        oldGammaC, oldLambdaC, oldGammaD, oldLambdaD = GammaC, LambdaC, GammaD, LambdaD
        if option == 'horizontal':
            GammaD, LambdaD, GammaC, LambdaC = myBmpsRowStep(GammaC, LambdaC, GammaD, LambdaD, envOpAB)
            GammaC, LambdaC, GammaD, LambdaD = myBmpsRowStep(GammaD, LambdaD, GammaC, LambdaC, envOpAB)
            # GammaC, LambdaC, GammaD, LambdaD = bmpsRowStep(GammaC, LambdaC, GammaD, LambdaD, envOpAB, option)
            # GammaD, LambdaD, GammaC, LambdaC = bmpsRowStep(GammaD, LambdaD, GammaC, LambdaC, envOpAB, option)
        elif option == 'diagonal':
            GammaC, LambdaC, GammaD, LambdaD = bmpsRowStep(GammaC, LambdaC, GammaD, LambdaD, AEnv, option)
            GammaD, LambdaD, GammaC, LambdaC = bmpsRowStep(GammaD, LambdaD, GammaC, LambdaC, BEnv, option)
        bops.removeState([oldGammaC, oldLambdaC, oldGammaD, oldLambdaD])
        convergence.append(checkConvergence(oldGammaC, oldLambdaC, oldGammaD, oldLambdaD, GammaC, LambdaC, GammaD, LambdaD, 2))

    # TODO compare density matrices as a convergence step. Find a common way to do this for iMPS.
    # TODO (for toric code, did it manually and it works)
    if option == 'diagonal':
        cUp = bops.multiContraction(GammaD, LambdaD, '3', '0')
        dUp = bops.multiContraction(GammaC, LambdaC, '3', '0')
        oldGammaC, oldLambdaC, oldGammaD, oldLambdaD = GammaC, LambdaC, GammaD, LambdaD
        GammaC, LambdaC, GammaD, LambdaD = bmpsRowStep(GammaC, LambdaC, GammaD, LambdaD, AEnv, option)
        cDown = bops.multiContraction(GammaD, LambdaD, '3', '0')
        dDown = bops.multiContraction(GammaC, LambdaC, '3', '0')
        bops.removeState([GammaC, LambdaC, GammaD, LambdaD, oldGammaC, oldLambdaC, oldGammaD, oldLambdaD])
        return cUp, cDown, dUp, dDown
    if option == 'horizontal':
        c = bops.multiContraction(GammaC, LambdaC, '2', '0')
        d = bops.multiContraction(GammaD, LambdaD, '2', '0')
        GammaC, LambdaC, GammaD, LambdaD = bmpsRowStep(GammaC, LambdaC, GammaD, LambdaD, envOpAB, option)
        cA = bops.multiContraction(GammaC, LambdaC, '2', '0')
        dB = bops.multiContraction(GammaD, LambdaD, '2', '0')
        bops.removeState([GammaC, LambdaC, GammaD, LambdaD, oldGammaC, oldLambdaC, oldGammaD, oldLambdaD])
        return c, d, cA, dB

# Start with a 2*2 DM, increase later
def bmpsCols(c, d, cA, dB, AEnv, BEnv, steps):
    envOpAB = bops.permute(bops.multiContraction(AEnv, BEnv, '1', '3'), [0, 3, 2, 4, 1, 5])
    envOpBA = bops.permute(bops.multiContraction(BEnv, AEnv, '1', '3'), [0, 3, 2, 4, 1, 5])
    row = bops.multiContraction(c, d, '2', '0') # C and D expect the reversed order of AB, but this is just some startup state anyway
    for i in range(steps):
        row = bops.multiContraction(bops.multiContraction(d, row, '2', '0', cleanOriginal1=True, cleanOriginal2=True),
                                    c, '4', '0', cleanOriginal1=True, cleanOriginal2=True)
        row = bops.permute(bops.multiContraction(row, envOpBA, '2314', '0123', cleanOriginal1=True, cleanOriginal2=True),
                           [0, 2, 3, 1])
        row = bops.multiContraction(bops.multiContraction(c, row, '2', '0', cleanOriginal1=True, cleanOriginal2=True),
                                    d, '4', '0', cleanOriginal1=True, cleanOriginal2=True)
        row = bops.permute(bops.multiContraction(row, envOpAB, '2314', '0123', cleanOriginal1=True, cleanOriginal2=True),
                           [0, 2, 3, 1])
        norm = np.sqrt(bops.multiContraction(row, row, '0123', '3210').tensor)
        row = bops.multNode(row, 1 / norm)
    return row


def bmpsDensityMatrix(c, d, cA, dB, AEnv, BEnv, A, B, steps):
    upRow = bmpsCols(c, d, cA, dB, AEnv, BEnv, steps)
    sideRow = bops.multiContraction(c, d, '2', '0')
    LShape = bops.multiContraction(sideRow, upRow, '3', '0', cleanOriginal1=True, cleanOriginal2=True)
    circle = bops.multiContraction(LShape, LShape, '05', '05', cleanOriginal1=True)
    ABNet = bops.permute(bops.multiContraction(bops.multiContraction(B, A, '2', '4'),
                                  bops.multiContraction(A, B, '2', '4'), '28', '16', cleanOriginal2=True, cleanOriginal1=True),
                         [1, 5, 6, 13, 14, 9, 10, 2, 0, 4, 8, 12, 3, 7, 11, 15])
    dm = bops.multiContraction(circle, ABNet, '01234567', '01234567', cleanOriginal1=True, cleanOriginal2=True)
    return dm



# def bmpsColStepA(X, cUp, cDown, dUp, dDown, AEnv, BEnv):
#     dx = bops.multiContraction(dUp, X, '3', '0')
#     dxd = bops.multiContraction(dx, dDown, '5', '3')
#     dxdB = bops.multiContraction(dxd, BEnv, '3467', '2367')
#     cdxdB = bops.multiContraction(cUp, dxdB, '3', '0')
#     cdxdBA = bops.multiContraction(cdxdB, AEnv, '123467', '012367')
#     final = bops.multiContraction(cdxdBA, cDown, '123', '312')
#     bops.removeState([dx, dxd, dxdB, cdxdB, cdxdBA])
#     return final
#
#
# def bmpsColStepB(X, cUp, cDown, dUp, dDown, AEnv, BEnv):
#     cx = bops.multiContraction(cDown, X, '0', '0')
#     cxc = bops.multiContraction(cx, cUp, '5', '0')
#     cxcA = bops.multiContraction(cxc, AEnv, '3456', '4501')
#     dcxcA = bops.multiContraction(dDown, cxcA, '0', '2')
#     dcxcAB = bops.multiContraction(dcxcA, BEnv, '893401', '014567')
#     final = bops.multiContraction(dcxcAB, dUp, '123', '012')
#     bops.removeState([cx, cxc, cxcA, dcxcA, dcxcAB])
#     return final
#
#
# def getBMPSSiteOps(steps, Xa, Xb, cUp, cDown, dUp, dDown, AEnv, BEnv):
#     xatests = [0] * steps
#     xbtests = [0] * steps
#     for i in range(steps):
#         xbForm = Xb
#         Xb = bmpsColStepB(Xb, cUp, cDown, dUp, dDown, AEnv, BEnv)
#         xaForm = Xa
#         Xa = bmpsColStepA(Xa, cUp, cDown, dUp, dDown, AEnv, BEnv)
#         norm = np.sqrt(bops.multiContraction(Xa, Xb, '0123', '3120').tensor)
#         Xa = bops.multNode(Xa, 1 / norm)
#         Xb = bops.multNode(Xb, 1 / norm)
#         xatests[i] = bops.multiContraction(xaForm, Xa, '0123', '0123*').tensor * 1
#         xbtests[i] = bops.multiContraction(xbForm, Xb, '0123', '0123*').tensor * 1
#         # dm = np.reshape(getDM(Xa, Xb).tensor, [4, 4])
#         # dmForm = np.reshape(getDM(xaForm, xbForm).tensor, [4, 4])
#         if i == steps - 2:
#             b = 1
#         tn.remove_node(xaForm)
#         tn.remove_node(xbForm)
#     return Xa, Xb

