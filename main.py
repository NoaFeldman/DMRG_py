import tensornetwork as tn
import numpy as np
import basicOperations as bops
import DMRG as dmrg
import experimantalRandom as exr
import pickle
import os
import sys
import shutil


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


for NA in [4 * i for i in range(1, 7)]:
    NB = NA
    N = NA + NB
    onsiteTermsXX, neighborTermsXX = getXXHamiltonianMatrices(1, 0)
    psi = bops.getStartupState(N)
    HXX = dmrg.getDMRGH(N, onsiteTermsXX, neighborTermsXX)
    HLs, HRs = dmrg.getHLRs(HXX, psi)
    psiXX, E0, truncErrs = dmrg.getGroundState(HXX, HLs, HRs, psi, None)
    with open('results/psiXX_NA_' + str(NA) + '_NB_' + str(NB), 'wb') as f:
        pickle.dump(psiXX, f)
    for n in [2, 3, 4]:
        with open('results/expected_MPS_NA_' + str(NA) + '_NB_' + str(NB) + '_n_' + str(n), 'wb') as f:
            pickle.dump(bops.getRenyiEntropy(psiXX, n, NA), f)
    print('Done with N = ' + str(NA))



ASize = int(sys.argv[1])
n = int(sys.argv[2])
option = sys.argv[3]
dir = sys.argv[4] + '/' + option + '_NA_' + str(ASize) + '_n_' + str(n)
if option == 'asym':
    weight = np.sqrt(1/3) # float(sys.argv[5])
    dir = dir + '_' + str(weight)
rep = sys.argv[5]

try:
    os.mkdir(dir)
except FileExistsError:
    pass

if option == 'h2':
    psi = bops.getTestState_halvesAsPair(ASize * 2)
elif option == 'maxEntangled':
    psi = bops.getTestState_maximallyEntangledHalves(ASize * 2)
elif option == 'pair':
    psi = bops.getTestState_pair(ASize * 2 + 1)
elif option == 'XX':
    with open(sys.argv[4] + '/psiXX', 'rb') as f:
        psi = pickle.load(f)
elif option == 'asym':
    psi = bops.getTestState_unequalTwoStates(ASize + 1, weight)

Sn = bops.getRenyiEntropy(psi, n, ASize)
mySum = 0
M = 1000
steps = int(2**(ASize * n) / 1) * 10
results = np.zeros(steps + 1, dtype=complex)
results[0] = Sn
for k in range(len(psi) - 1, ASize - 1, -1):
    psi = bops.shiftWorkingSite(psi, k, '<<')
for m in range(M * steps):
    # vs = [[np.array([np.exp(1j * np.pi * np.random.randint(4) / 2), np.exp(1j * np.pi * np.random.randint(4) / 2)]) \
    #            for alpha in range(ASize)] for copy in range(n)]
    vs = [[np.array(
        [np.exp(1j * np.pi * np.random.randint(4)), np.exp(1j * np.pi * np.random.randint(4))]) \
           for alpha in range(ASize)] for copy in range(n)]
    currEstimation = exr.singleMeasurement(psi, vs)
    mySum += currEstimation
    if m % M == M - 1:
        results[int(m / M) + 1] = mySum / M
        with open(dir + '/' + option + '_' + rep + '_m_' + str(m), 'wb') as f:
            pickle.dump(mySum / M, f)
        mySum = 0

organized = []
for file in os.listdir(dir):
    if '_m_' in file:
        with open(dir + '/' + file, 'rb') as f:
            organized.append(pickle.load(f))
with open(sys.argv[4] + '/' + 'organized_' + option + '_' + str(n) + '_' + str(ASize) + '_' + rep, 'wb') as f:
    pickle.dump(organized, f)
convergence = np.zeros(len(organized) - 1)
convergence[0] = organized[0]
for i in range(1, len(convergence)):
    convergence[i] = (convergence[i - 1] * i + organized[i]) / (i + 1)
with open(sys.argv[4] + '/' + 'converged_' + option + '_' + str(n) + '_' + str(ASize) + '_' + rep, 'wb') as f:
    pickle.dump(convergence, f)
shutil.rmtree(dir)