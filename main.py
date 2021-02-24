import tensornetwork as tn
import numpy as np
import basicOperations as bops
import DMRG as dmrg
import experimantalRandom as exr
import pickle
from matplotlib import pyplot as plt


sigmaX = np.zeros((2, 2))
sigmaX[0][1] = 1
sigmaX[1][0] = 1
sigmaZ = np.zeros((2, 2))
sigmaZ[0][0] = 1
sigmaZ[1][1] = -1
sigmaY = 1j * np.matmul(sigmaZ, sigmaX)


def getXXHamiltonianMatrices(J, JDelta):
    onsiteTerms = [np.eye(2) * 0] * N
    neighborTerms = \
        [J/2 * np.kron(sigmaX, sigmaX) + J/2 * np.kron(sigmaY, sigmaY) + JDelta * np.kron(sigmaZ, sigmaZ)] * (N-1)
    return onsiteTerms, neighborTerms


# onsiteTerm = Omega * sigmaX
# neighborTerm = GammaC * kron(sigmaX, sigmaX)
def getHAMatrices(N, C, Omega, delta):
    onsiteTerms = [Omega * sigmaX] * N
    for i in range(N):
        onsiteTerms[i] += Omega * np.random.normal(0, delta) * sigmaZ
    neighborTerms = [C * np.kron(sigmaX, sigmaX)] * (N-1)
    return onsiteTerms, neighborTerms


def projectS(psi, AEnd):
    projector0Tensor = np.zeros((2, 2))
    projector0Tensor[0, 0] = 1
    projector0 = tn.Node(projector0Tensor, backend=None)
    projector1Tensor = np.zeros((2, 2))
    projector1Tensor[1, 1] = 1
    projector1 = tn.Node(projector1Tensor, backend=None)
    for i in range(AEnd + 1):
        if i % 2 == 0:
            psi[i] = bops.permute(bops.multiContraction(psi[i], projector0, [1], [0]), [0, 2, 1])
        else:
            psi[i] = bops.permute(bops.multiContraction(psi[i], projector1, [1], [0]), [0, 2, 1])


def fullState(psi):
    curr = psi[0]
    for i in range(1, len(psi)):
        curr = bops.multiContraction(curr, psi[i], [i + 1], [0])
    ten = np.round(curr.tensor, decimals=5)
    return ten


N = 8
T = 1
C = 1/T
J = C
Omega = 1/T
delta = 1
onsiteTermsXX, neighborTermsXX = getXXHamiltonianMatrices(1, 0)
psi = bops.getStartupState(N)

HXX = dmrg.getDMRGH(N, onsiteTermsXX, neighborTermsXX)
HLs, HRs = dmrg.getHLRs(HXX, psi)
psi, E0, truncErrs = dmrg.getGroundState(HXX, HLs, HRs, psi, None)

import sys
n= int(sys.argv[1])
ASize int(sys.argv[2])
psiCurr = bops.copyState(psi)
print('ASize = ' + str(ASize))
print('n = ' + str(n))

Sn = bops.getRenyiEntropy(psiCurr, n, ASize - 1)
print('Sn = ' + str(Sn))
mySum = 0
M = 1000
steps = 100 * 2**ASize
results = np.zeros(steps + 1)
results[0] = Sn
# from datetime import datetime
for k in range(N - 1, ASize - 1, -1):
    psi = bops.shiftWorkingSite(psiCurr, k, '<<')
    # start = datetime.now()
    for m in range(M * steps):
        vs = [[np.array([np.random.randint(2) * 2 - 1, np.random.randint(2) * 2 - 1]) \
                   for alpha in range(ASize)] for copy in range(n)]
        mySum += exr.singleMeasurement(psiCurr, vs)
        if m % M == M - 1:
            results[int(m / M) + 1] = mySum / M
            mySum = 0
            # end = datetime.now()
with open('results/experimental_N_' + str(N) + '_NA_' + str(ASize) +'_n_' + str(n), 'wb') as f:
    pickle.dump(results, f)

