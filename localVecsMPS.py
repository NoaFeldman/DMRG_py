import tensornetwork as tn
import numpy as np
import basicOperations as bops
from typing import List
import pickle
import sys
import os


def localVecsEstimate(psi: List[tn.Node], vs: List[List[np.array]]):
    vs = np.round(vs, 10)
    n = len(vs)
    NA = len(vs[0])
    result = 1
    for copy in range(n):
        curr = bops.multiContraction(psi[NA], psi[NA], '12', '12*')
        psiCopy = bops.copyState(psi)
        for alpha in range(NA - 1, -1, -1):
            toEstimate = np.outer(vs[copy][alpha], np.conj(vs[np.mod(copy + 1, n)][alpha]))
            psiCopy[alpha] = bops.permute(bops.multiContraction(psiCopy[alpha], tn.Node(toEstimate), \
                                                   '1', '1'), [0, 2, 1])
            curr = bops.multiContraction(bops.multiContraction(psiCopy[alpha], curr, '2', '0', cleanOr2=True),
                                         psi[alpha], '12', '12*', cleanOr1=True)
            # psiCopy = bops.shiftWorkingSite(psiCopy, alpha, '<<')
        result *= np.trace(curr.tensor)
        tn.remove_node(curr)
        bops.removeState(psiCopy)
    return result

n = int(sys.argv[1])
NA = int(sys.argv[2])
rep = sys.argv[3]
homedir = sys.argv[4]
mydir = homedir + '/XX_MPS_NA_' + str(NA) + '_n_' + str(n)
try:
    os.mkdir(mydir)
except FileExistsError:
    pass

with open(homedir + '/psiXX_' + str(NA * 2), 'rb') as f:
    psi = pickle.load(f)
Sn = bops.getRenyiEntropy(psi, n, NA)
mySum = 0
M = 1000
steps = int(2**(NA * n) / 1)
for k in range(len(psi) - 1, NA - 1, -1):
    psi = bops.shiftWorkingSite(psi, k, '<<')
for m in range(M * steps):
    vs = [[np.array([np.exp(1j * np.pi * np.random.randint(4) / 2), np.exp(1j * np.pi * np.random.randint(4) / 2)]) \
               for alpha in range(NA)] for copy in range(n)]
    currEstimation = localVecsEstimate(psi, vs)
    mySum += currEstimation
    if m % M == M - 1:
        with open(mydir + '/NA_' + str(NA) + '_n_' + str(n) + '_' + rep + '_m_' + str(m), 'wb') as f:
            pickle.dump(mySum / M, f)
        mySum = 0
