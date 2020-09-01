from scipy import linalg
import numpy as np
import basicOperations as bops
import randomMeasurements as rm
import sys
import tensornetwork as tn

chi = 32

# https://arxiv.org/pdf/0711.3960.pdf
def getTheta1(gammaL, lambdaL, gammaR, lambdaR, envOp):
    # TODO not the most efficient chi-wise (two chi^3 instead of 1). fix
    L = bops.multiContraction(gammaL, lambdaL, '3', '0')
    R = bops.multiContraction(gammaR, lambdaR, '3', '0')
    pair = bops.multiContraction(L, R, '3', '0')
    Theta = bops.permute(bops.multiContraction(pair, envOp, '1234', '0123'), [0, 2, 3, 4, 5, 1])

    tn.remove_node(R)
    tn.remove_node(L)
    tn.remove_node(pair)

    return Theta


def getTheta2(gammaL, lambdaL, gammaR, lambdaR, envOp):
    # TODO not the most efficient chi-wise (two chi^3 instead of 1). fix
    L = bops.multiContraction(lambdaR, gammaL, '1', '0')
    R = bops.multiContraction(lambdaL, gammaR, '1', '0')
    pair = bops.multiContraction(L, R, '3', '0')
    Theta = bops.permute(bops.multiContraction(pair, envOp, '1234', '0123'), [0, 2, 3, 4, 5, 1])

    tn.remove_node(R)
    tn.remove_node(L)
    tn.remove_node(pair)

    return Theta


def getTheta(gammaL, lambdaMid, gammaR, envOp):
    L = bops.multiContraction(gammaL, lambdaMid, '3', '0')
    pair = bops.multiContraction(L, gammaR, '3', '0')
    Theta = bops.permute(bops.multiContraction(pair, envOp, '1234', '0123'), [0, 2, 3, 4, 5, 1])

    tn.remove_node(L)
    tn.remove_node(pair)

    return Theta


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
    res = tn.Node(vTensor, backend=None)
    norm = np.sqrt(bops.multiContraction(res,res,'01', '01*').tensor)
    return bops.multNode(res, 1 / norm), w


def bmpsRowStep(GammaL, LambdaL, GammaR, LambdaR, envOp):
    oldLambdaR = LambdaR
    theta1 = getTheta1(GammaL, LambdaL, GammaR, LambdaR, envOp)
    M = bops.multiContraction(theta1, theta1, '1234', '1234*')
    vR, w = largestEigenvector(M, '>>')
    [u, xvals, v, truncErr] = bops.svdTruncation(vR, leftEdges=[vR[0]], rightEdges=[vR[1]], dir='>*<')
    xvals.tensor = np.diag(np.sqrt(np.diag(xvals.tensor)))
    X = bops.multiContraction(u, xvals, '1', '0')
    xValsInverseTensor = np.diag([1 / val for val in np.diag(xvals.tensor)])
    xValsInverseTensor[np.isnan(xValsInverseTensor)] = 0
    xValsInverse = tn.Node(xValsInverseTensor, backend=None)
    XInverse = bops.multiContraction(xValsInverse, u, '1', '1*')
    tn.remove_node(v)
    tn.remove_node(xvals)
    tn.remove_node(xValsInverse)
    tn.remove_node(u)
    theta2 = getTheta2(GammaL, LambdaL, GammaR, LambdaR, envOp)
    M = bops.multiContraction(theta2, theta2, '1234', '1234*')
    vL, w = largestEigenvector(M, '<<')
    [u, yvals, v, truncErr] = bops.svdTruncation(vL, leftEdges=[vL[0]], rightEdges=[vL[1]], dir='>*<')
    yvals.tensor = np.diag(np.sqrt(np.diag(yvals.tensor)))
    Yt = bops.multiContraction(u, yvals, '1', '0')
    yValsInverseTensor = np.diag([1 / val for val in np.diag(yvals.tensor)])
    yValsInverseTensor[np.isnan(yValsInverseTensor)] = 0
    yValsInverse = tn.Node(yValsInverseTensor, backend=None)
    YtInverse = bops.multiContraction(yValsInverse, u, '1', '1*')
    tn.remove_node(v)
    tn.remove_node(yvals)
    tn.remove_node(yValsInverse)
    tn.remove_node(u)
    Yt[0] ^ LambdaR[0]
    LambdaR[1] ^ X[0]
    newLambda = tn.contract_between(tn.contract_between(Yt, LambdaR), X)
    [U, LambdaR, V, truncErr] = bops.svdTruncation(newLambda, [newLambda[0]], [newLambda[1]], dir='>*<', maxBondDim=chi)

    # LambdaR.tensor = np.round(LambdaR.tensor, 10)

    LambdaR = bops.multNode(LambdaR, 1 / np.sqrt(np.trace(np.power(LambdaR.tensor, 2))))
    theta = getTheta(GammaL, LambdaL, GammaR, envOp)
    bops.removeState([GammaL, GammaR])
    LambdaRLeft = bops.copyState([LambdaR])[0]
    LambdaRRight = bops.copyState([LambdaR])[0]
    LambdaRLeft[1] ^ V[0]
    V[1] ^ XInverse[0]
    XInverse[1] ^ theta[0]
    theta[5] ^ YtInverse[1]
    YtInverse[0] ^ U[0]
    U[1] ^ LambdaRRight[0]
    Sigma = tn.contract_between(tn.contract_between(tn.contract_between(tn.contract_between(tn.contract_between( \
                                tn.contract_between( \
                                LambdaRLeft, V), XInverse), \
                                theta), YtInverse), U), LambdaRRight)
    [P, LambdaL, Q, truncErr] = bops.svdTruncation(Sigma, Sigma[:3], Sigma[3:], dir='>*<', maxBondDim=chi)
    LambdaL = bops.multNode(LambdaL, 1 / np.sqrt(np.trace(np.power(LambdaL.tensor, 2))))
    lambdaInverseTensor = 1 / np.diag(LambdaRLeft.tensor)
    lambdaInverseTensor[np.isnan(lambdaInverseTensor)] = 0
    LambdaRLeft = tn.Node(np.diag(lambdaInverseTensor))
    LambdaRRight = tn.Node(np.diag(lambdaInverseTensor))
    LambdaRLeft[1] ^ P[0]
    GammaL = tn.contract_between(LambdaRLeft, P)
    Q[3] ^ LambdaRRight[0]
    GammaR = tn.contract_between(Q, LambdaRRight)


    # trio = bops.multiContraction(Yt, bops.multiContraction(oldLambdaR, X, '1', '0'), '1', '0')
    # isTrio = bops.multiContraction(U, bops.multiContraction(LambdaR, V, '1', '0'), '1', '0')
    # oldlbInverse = tn.Node(np.linalg.inv(oldLambdaR.tensor))
    # trioInv = bops.multiContraction(XInverse, bops.multiContraction(oldlbInverse, YtInverse, '1', '0'), '1', '0')
    # uInverse = tn.Node(np.linalg.inv(U.tensor))
    # vInverse = tn.Node(np.linalg.inv(V.tensor))
    # lambdaInverse = tn.Node(np.linalg.inv(LambdaR.tensor))
    # isTrioInv = bops.multiContraction(vInverse, bops.multiContraction(lambdaInverse, uInverse, '1', '0'), '1', '0')

    return GammaL, LambdaL, GammaR, LambdaR


def checkCannonization(GammaC, LambdaC, GammaD, LambdaD):
    c = bops.multiContraction(GammaC, LambdaC, '3', '0')
    id = np.round(bops.multiContraction(c, c, '123', '123*').tensor, 4)
    if not (np.all(np.diag(np.diag(id)) == id) and set(np.diag(id)).issubset({0, 1})):
    # if not np.all(id == np.eye(len(id), dtype=complex)):
        return False
    c = bops.multiContraction(LambdaD, GammaC, '1', '0')
    id = np.round(bops.multiContraction(c, c, '012', '012*').tensor, 4)
    if not (np.all(np.diag(np.diag(id)) == id) and set(np.diag(id)).issubset({0, 1})):
        return False
    c = bops.multiContraction(GammaD, LambdaD, '3', '0')
    id = np.round(bops.multiContraction(c, c, '123', '123*').tensor, 4)
    if not (np.all(np.diag(np.diag(id)) == id) and set(np.diag(id)).issubset({0, 1})):
        return False
    c = bops.multiContraction(LambdaC, GammaD, '1', '0')
    id = np.round(bops.multiContraction(c, c, '012', '012*').tensor, 4)
    if not (np.all(np.diag(np.diag(id)) == id) and set(np.diag(id)).issubset({0, 1})):
    # if not np.all(id == np.eye(len(id), dtype=complex)):
        return False
    return True


def checkConvergence(oldGammaC, oldLambdaC, oldGammaD, oldLambdaD, GammaC, LambdaC, GammaD, LambdaD, d):
    dmC = getRowDM(GammaC, LambdaC, GammaD, LambdaD, 1, d)
    oldDmC = getRowDM(oldGammaC, oldLambdaC, oldGammaD, oldLambdaD, 1, d)
    dmD = getRowDM(GammaD, LambdaD, GammaC, LambdaC, 1, d)
    oldDmD = getRowDM(oldGammaD, oldLambdaD, oldGammaC, oldLambdaC, 1, d)
    return max(map(max, dmC - oldDmC)) / max(map(max, dmC)), max(map(max, dmD - oldDmD)) / max(map(max, dmD))


def getRowDM(GammaL, LambdaL, GammaR, LambdaR, sites, d):
    c = bops.multiContraction(bops.multiContraction(LambdaR, GammaL, '1', '0'), LambdaL, '3', '0')
    row = bops.multiContraction(bops.multiContraction(c, GammaR, '3', '0'), LambdaR, '5', '0')
    for i in range(sites):
        row = bops.multiContraction(row, GammaL, [len(row.edges) - 1], [0])
        row = bops.multiContraction(row, LambdaL, [len(row.edges) - 1], [0])
        row = bops.multiContraction(row, GammaR, [len(row.edges) - 1], [0])
        row = bops.multiContraction(row, LambdaR, [len(row.edges) - 1], [0])
    dm = bops.multiContraction(row, row, [0, len(row.edges) - 1], [0, len(row.edges) - 1, '*'])
    return np.reshape(dm.tensor, [d**(2 * (2 + 2 * sites)), d**(2 * (2 + 2 * sites))])


def getBMPSRowOps(GammaC, LambdaC, GammaD, LambdaD, AEnv, BEnv, steps):
    for i in range(steps):
        oldGammaC, oldLambdaC, oldGammaD, oldLambdaD = GammaC, LambdaC, GammaD, LambdaD
        GammaC, LambdaC, GammaD, LambdaD = bmpsRowStep(GammaC, LambdaC, GammaD, LambdaD, AEnv)
        checkCannonization(GammaC, LambdaC, GammaD, LambdaD)
        GammaD, LambdaD, GammaC, LambdaC = bmpsRowStep(GammaD, LambdaD, GammaC, LambdaC, BEnv)
        checkCannonization(GammaC, LambdaC, GammaD, LambdaD)
        bops.removeState([oldGammaC, oldLambdaC, oldGammaD, oldLambdaD])

    # TODO compare density matrices as a convergence step. Find a common way to do this for iMPS.
    # TODO (for toric code, did it manually and it works)

    cUp = bops.multiContraction(GammaD, LambdaD, '3', '0')
    dUp = bops.multiContraction(GammaC, LambdaC, '3', '0')
    oldGammaC, oldLambdaC, oldGammaD, oldLambdaD = GammaC, LambdaC, GammaD, LambdaD
    GammaC, LambdaC, GammaD, LambdaD = bmpsRowStep(GammaC, LambdaC, GammaD, LambdaD, AEnv)
    cDown = bops.multiContraction(GammaD, LambdaD, '3', '0')
    dDown = bops.multiContraction(GammaC, LambdaC, '3', '0')
    bops.removeState([GammaC, LambdaC, GammaD, LambdaD, oldGammaC, oldLambdaC, oldGammaD, oldLambdaD])
    return cUp, cDown, dUp, dDown

#
# XbTensor = np.zeros((cDown[0].dimension, d, d, cUp[0].dimension))
# for i in range(len(XbTensor)):
#     for j in range(len(XbTensor[0, 0, 0])):
#         for s1 in range(d):
#             for s2 in range(d):
#                 XbTensor[i, s1,  s2, j] = (int(i == j) - int(i != j)) * (int(s1 == s2) - int(s1 !=s2))
# Xb = tn.Node(XbTensor) # just an initial guess with the propper dimensions
# norm = np.sqrt(bops.multiContraction(Xb, Xb, '0123', '0123*').tensor)
# Xb = bops.multNode(Xb, 1 / norm)
# XaTensor = np.zeros((dUp[3].dimension, d, d, dDown[3].dimension))
# for i in range(len(XaTensor)):
#     for j in range(len(XaTensor[0, 0, 0])):
#         for s1 in range(d):
#             for s2 in range(d):
#                 XaTensor[i, s1,  s2, j] = (int(i == j) - int(i != j)) * (int(s1 == s2) - int(s1 !=s2))
# Xa = tn.Node(XaTensor) # just an initial guess with the propper dimensions
# norm = np.sqrt(bops.multiContraction(Xa, Xa, '0123', '0123*').tensor)
# Xa = bops.multNode(Xa, 1 / norm)
#
# def bmpsColStepA(X):
#     dx = bops.multiContraction(dUp, X, '3', '0')
#     dxd = bops.multiContraction(dx, dDown, '5', '3')
#     dxdB = bops.multiContraction(dxd, BEnv, '3467', '2367')
#     cdxdB = bops.multiContraction(cUp, dxdB, '3', '0')
#     cdxdBA = bops.multiContraction(cdxdB, AEnv, '123467', '012367')
#     return bops.multiContraction(cdxdBA, cDown, '123', '312')
#
#
# def bmpsColStepB(X):
#     cx = bops.multiContraction(cDown, X, '0', '0')
#     cxc = bops.multiContraction(cx, cUp, '5', '0')
#     cxcA = bops.multiContraction(cxc, AEnv, '3456', '4501')
#     dcxcA = bops.multiContraction(dDown, cxcA, '0', '2')
#     dcxcAB = bops.multiContraction(dcxcA, BEnv, '893401', '014567')
#     final = bops.multiContraction(dcxcAB, dUp, '123', '012')
#     bops.removeState([cx, cxc, cxcA, dcxcA, dcxcAB])
#     norm = np.sqrt(bops.multiContraction(final, final, '0123', '0123*').tensor)
#     return bops.multNode(final, 1 / norm)
#
#
# def getDM(Xa, Xb):
#     XbC = bops.multiContraction(Xb, cUp, '3', '0')
#     XbCD = bops.multiContraction(XbC, dUp, '5', '0')
#     XbCDXa = bops.multiContraction(XbCD, Xa, '7', '0')
#     XbCDXaD = bops.multiContraction(XbCDXa, dDown, '9', '3')
#     circle = bops.multiContraction(XbCDXaD, cDown, [9, 0], '30')
#     AB = bops.permute(bops.multiContraction(A, B, '3', '0'), [2, 0, 1, 4, 6, 5, 3, 7])
#     circleAB = bops.multiContraction(circle, AB, [0, 2, 4, 6, 8, 10], '012345')
#     final = bops.multiContraction(circleAB, AB, '012345', '012345*')
#     bops.removeState([XbC, XbCD, XbCDXa, XbCDXaD, circle, AB, circleAB])
#     return final
#
#
# steps = 100
# xatests = [0] * steps
# xbtests = [0] * steps
# for i in range(steps):
#     xbForm = Xb
#     Xb = bmpsColStepB(Xb)
#     xaForm = Xa
#     Xa = bmpsColStepA(Xa)
#     norm = np.sqrt(bops.multiContraction(Xa, Xb, '0123', '3120').tensor)
#     Xa = bops.multNode(Xa, 1 / norm)
#     Xb = bops.multNode(Xb, 1 / norm)
#     xatests[i] = bops.multiContraction(xaForm, Xa, '0123', '0123*').tensor * 1
#     xbtests[i] = bops.multiContraction(xbForm, Xb, '0123', '0123*').tensor * 1
#     dm = np.reshape(getDM(Xa, Xb).tensor, [4, 4])
#     dmForm = np.reshape(getDM(xaForm, xbForm).tensor, [4, 4])
#     if i == steps - 2:
#         b = 1
#     tn.remove_node(xaForm)
#     tn.remove_node(xbForm)

