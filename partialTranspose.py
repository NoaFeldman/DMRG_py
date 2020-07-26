import scipy
import tensornetwork as tn
import numpy as np
import basicOperations as bops
import DMRG as dmrg

""" Technique here - https://arxiv.org/pdf/1605.00674.pdf """

def getEMatrix(psi, startInd, endInd):
    psiDagger = bops.copyState(psi, conj=True)
    E = bops.multiContraction(psi[startInd], psiDagger[startInd], [1], [1])
    for i in range(startInd + 1, endInd + 1):
        E = bops.multiContraction(E, bops.multiContraction(psi[i], psiDagger[i], [1], [1]), [1, 3], [0, 2])
    bops.removeState(psiDagger)
    return E


def getNMatrices(E, subsystemNum):
    [N, Nbar, truncErr] = bops.svdTruncation(E, [E[0], E[2]], [E[1], E[3]], dir='>>', edgeName='S' + str(subsystemNum))
    return N, Nbar


def getDensityMatrix(psi, A1Start, A1End, A2Start, A2End):
    psiCopy = bops.copyState(psi)
    for k in [len(psiCopy) - 1 - i for i in range(len(psiCopy) - A2End - 1)]:
        psiCopy = bops.shiftWorkingSite(psiCopy, k, '<<')
    E1 = getEMatrix(psiCopy, A1Start, A1End)
    bops.copyState([E1])
    N1, N1bar = getNMatrices(E1, 1)
    N1bar.edges[0].name += '*'
    E2 = getEMatrix(psiCopy, A2Start, A2End)
    N2, N2bar = getNMatrices(E2, 2)
    N2bar.edges[0].name += '*'
    res = bops.multiContraction(bops.multiContraction(N1, N1bar, [0], [1]),
                                bops.multiContraction(N2, N2bar, [1], [2]), [0, 3], [0, 3])
    bops.removeState(psiCopy)
    return res


def getPartiallyTransposed(psi, A1Start, A1End, A2Start, A2End):
    res = getDensityMatrix(psi, A1Start, A1End, A2Start, A2End)
    res = bops.permute(res, [0, 3, 1, 2])
    return res


def getStraightDM(psi, A1Start, A1End, A2Start, A2End):
    res = getDensityMatrix(psi, A1Start, A1End, A2Start, A2End)
    res = bops.permute(res, [0, 2, 1, 3])
    return res
