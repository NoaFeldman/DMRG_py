import tensornetwork as tn
import numpy as np
import basicOperations as bops
from typing import List
import pickle
import sys
import os
import randomUs as ru


def localVecsEstimate(psi: List[tn.Node], vs: List[List[np.array]], half='left'):
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
mydir = homedir + '/XX_MPS_NA_' + str(NA) + '_NB_' + str(NB) + '_n_' + str(n) + '_optimized'
try:
    os.mkdir(mydir)
except FileExistsError:
    pass
theta = np.pi / 5
phi = np.pi / 5
U = tn.Node(np.matmul(ru.getUPhi(np.pi * phi / 2, 2), ru.getUTheta(np.pi * theta / 2, 2)))

with open(homedir + '/psiXX_NA_' + str(NA) + '_NB_' + str(NB), 'rb') as f:
    psi = pickle.load(f)
Sn = bops.getRenyiEntropy(psi, n, NA)
with open(homedir + '/expected_MPS_NA_' + str(NA) + '_NB_' + str(NB) + '_n_' + str(n) + '_' + half, 'wb') as f:
    pickle.dump(Sn, f)
mySum = 0
M = 1000
if half == 'left':
    steps = int(2 ** (NA * n))
    for k in range(len(psi) - 1, NA - 1, -1):
        psi = bops.shiftWorkingSite(psi, k, '<<')
    sites = range(NA)
else:
    steps = int(2 ** (NB * n))
    sites = range(NA, len(psi))
for site in sites:
    psi[site] = bops.permute(bops.multiContraction(psi[site], U, '1', '0'), [0, 2, 1])
for m in range(M * steps):
    if half == 'left':
        vs = [[np.array([np.exp(1j * np.pi * np.random.randint(4) / 2), np.exp(1j * np.pi * np.random.randint(4) / 2)]) \
               for alpha in range(NA)] for copy in range(n)]
    else:
        vs = [[np.array([np.exp(1j * np.pi * np.random.randint(4) / 2), np.exp(1j * np.pi * np.random.randint(4) / 2)]) \
               for alpha in range(NB)] for copy in range(n)]
    u = np.matmul(ru.getUTheta(theta, d=2), ru.getUPhi(phi, d=2))
    for i in range(len(vs)):
        for j in range(len(vs[0])):
            vs[i][j] = np.matmul(u, vs[i][j])
    currEstimation = localVecsEstimate(psi, vs, half=half)
    mySum += currEstimation
    if m % M == M - 1:
        with open(mydir + '/NA_' + str(NA) + '_n_' + str(n) + '_' + rep + '_' + half + '_m_' + str(m), 'wb') as f:
            pickle.dump(mySum / M, f)
        print('+')
        print(np.real(np.round(mySum / M, 3)))
        mySum = 0
