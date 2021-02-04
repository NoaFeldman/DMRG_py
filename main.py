import tensornetwork as tn
import numpy as np
import basicOperations as bops
import DMRG as dmrg
import trotter
import test
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


N = 16
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
print('E0 = ' + str(E0))
print('E0 = ' + str(dmrg.stateEnergy(psi, HXX)))
R2 = bops.getRenyiEntropy(psi, 2, int(len(psi) / 2 - 1))
print(R2)
