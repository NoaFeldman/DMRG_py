from scipy import linalg
import numpy as np
import basicOperations as bops
import randomMeasurements as rm
import sys
import tensornetwork as tn

d =  2

# Toric code model matrices - figure 30 here https://arxiv.org/pdf/1306.2164.pdf
ATensor = np.zeros((2, 2, 2, 2, 2))
ATensor[0, 0, 0, 0, 0] = 1
ATensor[1, 1, 1, 1, 0] = 1
ATensor[0, 0, 1, 1, 1] = 1
ATensor[1, 1, 0, 0, 1] = 1
A = tn.Node(ATensor, name='A', backend=None)
BTensor = np.zeros((2, 2, 2, 2, 2))
BTensor[0, 0, 0, 0, 0] = 1
BTensor[1, 1, 1, 1, 0] = 1
BTensor[0, 1, 1, 0, 1] = 1
BTensor[1, 0, 0, 1, 1] = 1
B = tn.Node(BTensor, name='B', backend=None)

AEnv = bops.permute(bops.multiContraction(A, A, '4', '4*'), [0, 4, 1, 5, 2, 6, 3, 7])
BEnv = bops.permute(bops.multiContraction(B, B, '4', '4*'), [0, 4, 1, 5, 2, 6, 3, 7])
chi = 32
# Double 'physical' leg for the closed MPS
GammaTensor = np.zeros((2, 2, 2, 2), dtype=complex)
GammaTensor[0, 1, 0, 1] = 1
GammaTensor[1, 0, 1, 0] = 1j
GammaTensor[0, 0, 0, 1] = 1j
GammaTensor[1, 1, 1, 0] = 1
GammaC = tn.Node(GammaTensor, name='GammaC', backend=None)
LambdaC = tn.Node(np.eye(2) / np.sqrt(2), backend=None)
GammaD = tn.Node(GammaTensor, name='GammaD', backend=None)
LambdaD = tn.Node(np.eye(2) / np.sqrt(2), backend=None)


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


def squareM(M):
    if M.edges[0].dimension == M.edges[1].dimension:
        return M
    elif M.edges[0].dimension > M.edges[1].dimension:
        temp = np.zeros((M.edges[0].dimension, M.edges[0].dimension, M.edges[0].dimension, M.edges[0].dimension))
        temp[:, :M.edges[1].dimension, :, :M.edges[1].dimension] = M.tensor
        M.tensor = temp
        return M
    else:
        temp = np.zeros((M.edges[1].dimension, M.edges[1].dimension, M.edges[1].dimension, M.edges[1].dimension))
        temp[:M.edges[0].dimension, :, :M.edges[0].dimension, :] = M.tensor
        M.tensor = temp
        return M


def getTridiagonal(M, dir):
    M = squareM(M)
    dim = M.edges[0].dimension
    betas = []
    alphas = []
    base = []
    counter = 0
    accuracy = 1e-10
    beta = 1
    formBeta = 2
    while beta > accuracy and counter < 50 and beta < formBeta:
        if counter == 0:
            vTensor = np.eye(dim) / np.sqrt(dim)
            v = tn.Node(vTensor, backend=None)
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

        w = Mv - alphaV
        formBeta = beta
        beta = np.sqrt(bops.multiContraction(w, w, '01', '01*').tensor)
        betas.append(beta)
        counter += 1
    # TODO clean up
    return alphas, betas, base


def lanczos(theta, dir):
    alphas, betas, base = getTridiagonal(theta, dir)
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


def bmpsRowStep(GammaL, LambdaL, GammaR, LambdaR, envOp):
    theta1 = getTheta1(GammaL, LambdaL, GammaR, LambdaR, envOp)
    M = bops.multiContraction(theta1, theta1, '1234', '1234*')
    vR = lanczos(M, '<<')
    [X, Xd, truncErr] = bops.svdTruncation(vR, leftEdges=[vR[0]], rightEdges=[vR[1]], dir='><')
    tn.remove_node(Xd)
    theta2 = getTheta2(GammaL, LambdaL, GammaR, LambdaR, envOp)
    M = bops.multiContraction(theta2, theta2, '1234', '1234*')
    vL = lanczos(M, '>>')
    [Yt, Ytd, truncErr] = bops.svdTruncation(vL, leftEdges=[vL[0]], rightEdges=[vL[1]], dir='><')
    tn.remove_node(Ytd)
    Yt[1] ^ LambdaR[0]
    LambdaR[1] ^ X[0]
    newLambda = tn.contract_between(tn.contract_between(Yt, LambdaR), X)
    [U, LambdaR, V, truncErr] = bops.svdTruncation(newLambda, [newLambda[0]], [newLambda[1]], dir='>*<', maxBondDim=chi)
    LambdaR = bops.multNode(LambdaR, 1 / np.sqrt(np.trace(np.power(LambdaR.tensor, 2))))
    theta = getTheta(GammaL, LambdaL, GammaR, envOp)
    bops.removeState([GammaL, GammaR])
    LambdaRLeft = bops.copyState([LambdaR])[0]
    LambdaRRight = bops.copyState([LambdaR])[0]
    XInverse = bops.copyState([X])[0]
    XInverse.tensor = np.linalg.inv(XInverse.tensor)
    YtInverse = bops.copyState([Yt])[0]
    YtInverse.tensor = np.linalg.inv(YtInverse.tensor)
    LambdaRLeft[1] ^ V[0]
    V[1] ^ XInverse[0]
    XInverse[1] ^ theta[0]
    theta[5] ^ YtInverse[0]
    YtInverse[1] ^ U[0]
    U[1] ^ LambdaRRight[0]
    Sigma = tn.contract_between(tn.contract_between(tn.contract_between(tn.contract_between(tn.contract_between( \
                                tn.contract_between( \
                                LambdaRLeft, V), XInverse), \
                                theta), YtInverse), U), LambdaRRight)
    [P, LambdaL, Q, truncErr] = bops.svdTruncation(Sigma, Sigma[:3], Sigma[3:], dir='>*<', maxBondDim=chi)
    LambdaL = bops.multNode(LambdaL, 1 / np.sqrt(np.trace(np.power(LambdaL.tensor, 2))))
    LambdaRLeft = bops.copyState([LambdaR])[0]
    LambdaRLeft.tensor = np.diag(1 / np.diag(LambdaRLeft.tensor))
    LambdaRRight = bops.copyState([LambdaR])[0]
    LambdaRRight.tensor = LambdaRLeft.tensor
    LambdaRLeft[1] ^ P[0]
    GammaL = tn.contract_between(LambdaRLeft, P)
    Q[3] ^ LambdaRRight[0]
    GammaR = tn.contract_between(Q, LambdaRRight)
    return GammaL, LambdaL, GammaR, LambdaR


def checkCannonization(GammaC, LambdaC, GammaD, LambdaD):
    cc = bops.multiContraction(GammaC, LambdaC, '3', '0')
    Lc = bops.multiContraction(cc, cc, '123', '123*')
    if np.amax(np.round(Lc.tensor / Lc.tensor[0, 0], 4) - np.eye(len(Lc.tensor))) > 0:
        return False
    # dd = bops.multiContraction(GammaD, LambdaD, '3', '0')
    # Ld = bops.multiContraction(dd, dd, '123', '123*')
    # if np.amax(np.round(Ld.tensor / Ld.tensor[0, 0], 4) - np.eye(len(Ld.tensor))) > 0:
    #     return False
    cd = bops.multiContraction(LambdaC, GammaD, '1', '0')
    Rcd = bops.multiContraction(cd, cd, '012', '012*')
    if np.amax(np.round(Rcd.tensor / Rcd.tensor[0, 0], 4) - np.eye(len(Rcd.tensor))) > 0:
        return False
    # dc = bops.multiContraction(LambdaD, GammaC, '1', '0')
    # Rdc = bops.multiContraction(dc, dc, '012', '012*')
    # if np.amax(np.round(Rdc.tensor / Rdc.tensor[0, 0], 4) - np.eye(len(Rdc.tensor))) > 0:
    #     return False
    return True


def checkConvergence(oldGammaC, oldLambdaC, oldGammaD, oldLambdaD, GammaC, LambdaC, GammaD, LambdaD):
    C = bops.multiContraction(GammaC, LambdaC, '3', '0')
    D = bops.multiContraction(LambdaC, bops.multiContraction(GammaD, LambdaD, '3', '0'), '1', '0')
    oldC = bops.multiContraction(oldGammaC, oldLambdaC, '3', '0')
    oldD = bops.multiContraction(oldLambdaC, bops.multiContraction(oldGammaD, oldLambdaD, '3', '0'), '1', '0')
    Cs = bops.multiContraction(C, oldC, '123', '123*')
    Ds = bops.multiContraction(D, oldD, '012', '012*')
    return bops.multiContraction(Cs, Ds, '01', '01').tensor * 1


steps = 50
ctests = [0] * steps
dtests = [0] * steps

c = bops.multiContraction(GammaC, LambdaC, '3', '0')
id = bops.multiContraction(c, c, '123', '123*')
c = bops.multiContraction(LambdaD, GammaC, '1', '0')

def getDM(GammaD, LambdaD, GammaC, LambdaC):
    c = bops.multiContraction(bops.multiContraction(LambdaD, GammaC, '1', '0'), LambdaC, '3', '0')
    row = bops.multiContraction(bops.multiContraction(c, GammaD, '3', '0'), LambdaD, '5', '0')
    dm = bops.multiContraction(row, row, '05', '05*')
    return np.reshape(dm.tensor, [16, 16])


for i in range(steps):
    oldGammaC, oldLambdaC, oldGammaD, oldLambdaD = GammaC, LambdaC, GammaD, LambdaD
    # GammaC, LambdaC, GammaD, LambdaD = bmpsRowStep(GammaC, LambdaC, GammaD, LambdaD, AEnv)
    GammaD, LambdaD, GammaC, LambdaC = bmpsRowStep(GammaD, LambdaD, GammaC, LambdaC, BEnv)
    if i > 5:
        ctests[i] = round(checkConvergence(oldGammaC, oldLambdaC, oldGammaD, oldLambdaD, GammaC, LambdaC, GammaD, LambdaD), 2)
        dtests[i] = round(sum(np.diag(oldLambdaD.tensor) * np.diag(LambdaD.tensor)), 2)
    if i == 98:
        b = 1


# TODO compare density matrices as a convergence step. Find a common way to do this for iMPS.
# TODO (for toric code, did it manually and it works)

D = bops.multiContraction(GammaD, LambdaD, '3', '0')
C = bops.multiContraction(GammaC, LambdaC, '3', '0')

XbTensor = np.zeros((D[0].dimension, d, d, D[0].dimension))
for i in range(D[0].dimension):
    for j in range(D[0].dimension):
        for s1 in range(d):
            for s2 in range(d):
                XbTensor[i, s1,  s2, j] = np.random.randint(2)
Xb = tn.Node(XbTensor, backend=None) # just an initial guess with the propper dimensions
norm = np.sqrt(bops.multiContraction(Xb, Xb, '0123', '0123*').tensor)
Xb = bops.multNode(Xb, 1 / norm)
Xa = tn.Node(Xb.get_tensor(), backend=None) # just an initial guess with the propper dimensions
norm = np.sqrt(bops.multiContraction(Xa, Xa, '0123', '0123*').tensor)
Xa = bops.multNode(Xa, 1 / norm)


def bmpsColStepB(X):
    xD = bops.multiContraction(X, D, '3', '0*')
    xDB = bops.multiContraction(xD, BEnv, '1234', '0123')
    xDBC = bops.multiContraction(xDB, C, '145', '012*')
    dXDBC = bops.multiContraction(D, xDBC, '0', '0')
    dXDBAC = bops.multiContraction(AEnv, dXDBC, '0123', '0134')
    final = bops.multiContraction(C, dXDBAC, '012', '401')
    bops.removeState([xD, xDB, xDBC, dXDBC, dXDBAC])
    final = bops.multNode(final, 1 / np.sqrt(bops.multiContraction(final, final, '0123', '0123*').tensor))
    return final


def bmpsColStepA(X):
    cx = bops.multiContraction(C, X, '3', '3')
    cxA = bops.multiContraction(cx, AEnv, '1245', '4567')
    cxAd = bops.multiContraction(cxA, D, '023', '312')
    cxAdc = bops.multiContraction(cxAd, C, '0', '3*')
    cxAdcB = bops.multiContraction(cxAdc, BEnv, '0145', '4567')
    final = bops.multiContraction(cxAdcB, D, '145',  '312*')
    final = bops.multNode(final, 1 / np.sqrt(bops.multiContraction(final, final, '0123', '0123*').tensor))
    bops.removeState([cx, cxA, cxAd, cxAdc, cxAdcB])
    return final


xatests = [0] * steps
xbtests = [0] * steps
for i  in range(steps):
    xbForm = Xb
    Xb = bmpsColStepB(Xb)
    xbtests[i] = bops.multiContraction(xbForm, Xb, '0123', '0123*').tensor * 1
    tn.remove_node(xbForm)
    xaForm = Xa
    Xa = bmpsColStepB(Xa)
    xatests[i] = bops.multiContraction(xaForm, Xa, '0123', '0123*').tensor * 1
    tn.remove_node(xaForm)

b = 1