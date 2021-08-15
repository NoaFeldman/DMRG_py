import tensornetwork as tn
import numpy as np
import basicOperations as bops
from typing import List
import pickle
import sys
import os
import randomUs as ru


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
        if option == 'flux':
            for alpha in sites:
                psiCopy[alpha] = bops.permute(bops.multiContraction(psiCopy[alpha], fourierTensor, '1', '0'), [0, 2, 1])

        # dual = bops.multiContraction(psi[0], psi[1], '2', '0')
        # triple = bops.multiContraction(dual, psi[2], '3', '0')
        # quaple = bops.multiContraction(triple, psi[3], '4', '0')
        # dualCopy = bops.multiContraction(psiCopy[0], psiCopy[1], '2', '0')
        # tripleCopy = bops.multiContraction(dualCopy, psiCopy[2], '3', '0')
        # quapleCopy = bops.multiContraction(tripleCopy, psiCopy[3], '4', '0')
        # dm = np.reshape(bops.multiContraction(quaple, quaple, '05', '05*').tensor, [16, 16])
        # phase = np.exp(1j * alpha)
        # fourierFull = np.diag([phase**(-4), phase**(-2), phase**(-2), 1, phase**(-2), 1, 1, phase**2, phase**(-2),
        #                        1, 1, phase**2, 1, phase**2, phase**2, phase**4])
        # dmftest = np.round(np.matmul(dm, np.conj(fourierFull)), 5)
        # dmf = np.round(np.reshape(bops.multiContraction(quaple, quapleCopy, '05', '05*').tensor, [16, 16]), 5)
        # idx0 = np.array((3, 5, 6, 9, 10, 12)).reshape(6, 1)
        # dm0 = dm[idx0, idx0.T]
        # p0 = sum(np.linalg.eigh(dm0)[0] ** 3)
        # idx2 = np.array((1, 2, 4, 8)).reshape(4, 1)
        # dm2 = dm[idx2, idx2.T]
        # p2 = sum(np.linalg.eigh(dm2)[0] ** 3)

        for alpha in sites:
            toEstimate = np.outer(vs[copy][alpha - NA], np.conj(vs[np.mod(copy + 1, n)][alpha - NA]))
            psiCopy[alpha] = bops.permute(bops.multiContraction(psiCopy[alpha], tn.Node(toEstimate), \
                                                   '1', '1'), [0, 2, 1])
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

n = int(sys.argv[1])
NA = int(sys.argv[2])
NB = NA
half = 'left'
rep = sys.argv[3]
homedir = sys.argv[4]
with open(homedir + '/psiXX_NA_' + str(NA) + '_NB_' + str(NB), 'rb') as f:
    psi = pickle.load(f)
option = sys.argv[5]
mydir = homedir + '/XX_MPS_NA_' + str(NA) + '_NB_' + str(NB) + '_n_' + str(n) + '_optimized'
if option == 'flux':
    alphaInd = int(sys.argv[6])
    alpha = np.pi * alphaInd / NA
    mydir += '_flux_' + str(alphaInd)
    fourierOp = np.eye(2, dtype=complex)
    fourierOp[0, 0] *= np.exp(-1j * alpha)
    fourierOp[1, 1] *= np.exp(1j * alpha)
    fourierTensor = tn.Node(fourierOp)
try:
    os.mkdir(mydir)
except FileExistsError:
    pass
theta = np.pi / 5
phi = np.pi / 5
U = tn.Node(np.matmul(ru.getUPhi(np.pi * phi / 2, 2), ru.getUTheta(np.pi * theta / 2, 2)))

Sn = bops.getRenyiEntropy(psi, n, NA)
print(Sn)
with open(homedir + '/expected_MPS_NA_' + str(NA) + '_NB_' + str(NB) + '_n_' + str(n) + '_' + half, 'wb') as f:
    pickle.dump(Sn, f)
mySum = 0
M = 1000
if half == 'left':
    steps = 500 # int(2 ** (NA * n))
    for k in range(len(psi) - 1, NA - 1, -1):
        psi = bops.shiftWorkingSite(psi, k, '<<')
    sites = range(NA)
else:
    steps = int(2 ** (NB * n))
    sites = range(NA, len(psi))
for m in range(M * steps):
    if half == 'left':
        vs = [[np.array([np.exp(1j * np.pi * np.random.randint(4) / 2), np.exp(1j * np.pi * np.random.randint(4) / 2)]) \
               for alpha in range(NA)] for copy in range(n)]
    else:
        vs = [[np.array([np.exp(1j * np.pi * np.random.randint(4) / 2), np.exp(1j * np.pi * np.random.randint(4) / 2)]) \
               for alpha in range(NB)] for copy in range(n)]
    for copy in range(n):
        for alpha in range(NA):
            vs[copy][alpha] = np.matmul(U.tensor, vs[copy][alpha])
    currEstimation = localVecsEstimate(psi, vs, option=option, half=half)
    mySum += currEstimation
    if m % M == M - 1:
        with open(mydir + '/NA_' + str(NA) + '_n_' + str(n) + '_' + rep + '_' + half + '_m_' + str(m), 'wb') as f:
            pickle.dump(mySum / M, f)
        print('+')
        print(np.round(mySum / M, 3))
        mySum = 0
