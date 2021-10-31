import tensornetwork as tn
import numpy as np
import basicOperations as bops
from typing import List
import pickle
import sys
import os
import randomUs as ru


def randomRenyiForDM(rho: List[tn.Node], n, outdir, rep):
    M = 1000
    NA = len(rho)
    mySum = 0
    steps = 2**(NA * n)
    for step in range(steps):
        for m in range(M):
            vs = getVs(n, len(rho), d=rho[0].edges[3].dimension)
            estimation = 1
            for copy in range(n):
                curr = tn.Node(np.eye(1))
                for site in range(NA):
                    curr = bops.multiContraction(curr, rho[site], '1', '0', cleanOr1=True)
                    toEstimate = tn.Node(np.outer(vs[copy][site - NA], np.conj(vs[np.mod(copy + 1, n)][site - NA])))
                    curr = bops.multiContraction(curr, toEstimate, '23', '01', cleanOr1=True, cleanOr2=True)
                estimation *= curr.tensor[0, 0]
            mySum += estimation
        with open(outdir + '/est_' + str(n) + '_' + str(NA) + '_' + str(rep), 'wb') as f:
            pickle.dump(mySum / M, f)
            print(mySum / M)
            mySum = 0


def localVecsEstimate(psi: List[tn.Node], vs: List[List[np.array]], option='', half='left'):
    vs = np.round(vs, 10)
    n = len(vs)
    result = 1
    for copy in range(n):
        if half == 'left':
            NA = len(vs[0])
            curr = bops.multiContraction(psi[NA], psi[NA], '12', '12*')
            sites = range(NA - 1, -1, -1)
        elif half == 'right':
            NA = len(psi) - len(vs[0])
            curr = bops.multiContraction(psi[len(psi) - NA - 1], psi[len(psi) - NA - 1], '01', '01*')
            sites = range(NA, len(psi))
        psiCopy = bops.copyState(psi)
        if option == 'flux' and copy == 0:
            for alpha in sites:
                psiCopy[alpha] = bops.multiContraction(psiCopy[alpha], fourierTensor, '1', '0').reorder_axes([0, 2, 1])
        for alpha in sites:
            toEstimate = np.outer(vs[copy][alpha - NA], np.conj(vs[np.mod(copy + 1, n)][alpha - NA]))
            psiCopy[alpha] = bops.multiContraction(psiCopy[alpha], tn.Node(toEstimate), '1', '1').reorder_axes([0, 2, 1])
            if half == 'left':
                curr = bops.multiContraction(bops.multiContraction(psiCopy[alpha], curr, '2', '0', cleanOr2=True),
                                         psi[alpha], '12', '12*', cleanOr1=True)
            elif half == 'right':
                curr = bops.multiContraction(bops.multiContraction(curr, psiCopy[alpha], '0', '0', cleanOr2=True),
                                         psi[alpha], '01', '01*', cleanOr1=True)
            # psiCopy = bops.shiftWorkingSite(psiCopy, alpha, '<<')
        result *= np.trace(curr.tensor)
        tn.remove_node(curr)
        bops.removeState(psiCopy)
    return result


vsPool = [np.array(arr) for arr in [[np.sqrt(2), 0], [0, np.sqrt(2)], [1, 1], [1, -1], [1, 1j], [1, -1j]]]
vsPool = vsPool + [-arr for arr in vsPool] + [1j * arr for arr in vsPool] + [-1j * arr for arr in vsPool]

def getVs(n, NA, half='left', option='', d=2, phi=0, theta=0, NB=0):
    if half == 'left':
        vs = [[vsPool[np.random.randint(len(vsPool))]
                   for alpha in range(NA)] for copy in range(n)]
        # vs = [[np.array([np.exp(1j * np.pi * np.random.randint(4) / 2) for i in range(d)])
        #            for alpha in range(NA)] for copy in range(n)]
    else:
        vs = [[np.array([np.exp(1j * np.pi * np.random.randint(4) / 2) for i in range(d)])
               for alpha in range(NB)] for copy in range(n)]
    if option == 'optimized':
        U = tn.Node(np.matmul(ru.getUPhi(np.pi * phi / 2, 2), ru.getUTheta(np.pi * theta / 2, 2)))
        for copy in range(n):
            for alpha in range(NA):
                vs[copy][alpha] = np.matmul(U.tensor, vs[copy][alpha])
    return vs


def getRandomizedRenyi(psi, n, NA, M, outdir, rep, option='optimized', half='left', theta=0, phi=0):
    mySum = 0
    if half == 'left':
        steps = int(2 ** (NA * n))
        for k in range(len(psi) - 1, NA - 1, -1):
            psi = bops.shiftWorkingSite(psi, k, '<<')
    else:
        NB = NA
        steps = int(2 ** (NB * n))
    for m in range(M * steps):
        # vs = getVs(n, NA, theta=theta, phi=phi)
        U = np.matmul(ru.getUPhi(np.pi * phi / 2, 2), ru.getUTheta(np.pi * theta / 2, 2))
        vsBasics = [np.matmul(U, np.array([0, 1])) * np.sqrt(2), np.matmul(U, np.array([1, 0])) * np.sqrt(2)]
        vs = [[vsBasics[np.random.randint(2)] for alpha in range(NA)] for copy in range(n)]
        currEstimation = localVecsEstimate(psi, vs, option=option, half=half)
        mySum += currEstimation
        if m % M == M - 1:
            with open(outdir + '/NA_' + str(NA) + '_n_' + str(n) + '_' + rep + '_' + half + '_m_' + str(m), 'wb') as f:
                pickle.dump(mySum / M, f)
            mySum = 0


def XXProcess(NA, n, flux, t=0.0, p=0.0, rep='1', indir='results', option='flux'):
    theta = t * np.pi
    phi = p * np.pi
    NB = NA
    M = 1000
    with open(indir + '/psiXX_NA_' + str(NA) + '_NB_' + str(NB), 'rb') as f:
        psi = pickle.load(f)
    mydir = indir + '/XX_MPS_NA_' + str(NA) + '_NB_' + str(NB) + '_n_' + str(n)
    if t != 0 or p != 0:
        mydir += '_' + str(np.round(t, 1)) + '_' + str(np.round(p, 1))
    if option == 'flux':
        alphaInd = flux
        alpha = np.pi * alphaInd / NA
        mydir += '_flux_' + str(alphaInd)
        fourierOp = np.eye(2, dtype=complex)
        fourierOp[0, 0] *= np.exp(-1j * alpha)
        fourierOp[1, 1] *= np.exp(1j * alpha)
        fourierTensor = tn.Node(fourierOp)
        theta = np.pi * t
        phi = np.pi * p
        mydir += '_t_' + str(t) + '_p_' + str(p)
    if option == 'optimized':
        theta = np.pi / 5
        phi = np.pi / 5
        mydir += '_optimized'
    try:
        os.mkdir(mydir)
    except FileExistsError:
        pass
    getRandomizedRenyi(psi, n, NA, M, mydir, rep, theta=theta, phi=phi)


NA = int(sys.argv[1])
n = int(sys.argv[2])
rep = sys.argv[3]
indir = sys.argv[4]
XXProcess(NA, n, flux=0, rep=rep, indir=indir, option='')