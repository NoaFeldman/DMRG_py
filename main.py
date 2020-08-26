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


def getXXZHamiltonianMatrices(J, JDelta):
    onsiteTerms = [np.eye(2)] * N
    neighborTerms = \
        [J * np.kron(sigmaX, sigmaX) + J * np.kron(sigmaY, sigmaY) + JDelta * np.kron(sigmaZ, sigmaZ)] * (N-1)
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
onsiteTermsXXZ, neighborTermsXXZ = getXXZHamiltonianMatrices(0, 1)
psi = bops.getStartupState(N, 'antiferromagnetic')

# HXXZ = dmrg.getDMRGH(N, onsiteTermsXXZ, neighborTermsXXZ)
# HLs, HRs = dmrg.getHLRs(HXXZ, psi0)
# psi, E0, truncErrs = dmrg.getGroundState(HXXZ, HLs, HRs, psi0, None)
# print('E0 = ' + str(E0))
# print('E0 = ' + str(dmrg.stateEnergy(psi, HXXZ)))
# hpsi = dmrg.applyH(psi, HXXZ)
# hpsi[0].tensor /= math.sqrt(abs(bops.getOverlap(hpsi, hpsi)))
# print('<psi|hpsi> = ' + str(bops.getOverlap(psi, hpsi)))
R2 = bops.getRenyiEntropy(psi, 2, int(len(psi) / 2 - 1))

NU = 200
etas = [1, 2, 3, N, int(N * 1.5), N * 2, int(N * 2.5), N * 3]
results = [None] * len(etas)
dt = J * 1e-2
for e in range(len(etas)):
    eta = etas[e]
    sum = 0
    for n in range(NU):
        psiCopy = bops.copyState(psi)
        for j in range(eta):
            onsiteTermsA, neighborTermsA = getHAMatrices(N, C, Omega, delta)
            HA = dmrg.getDMRGH(N, onsiteTermsA, neighborTermsA)
            trotterGates = trotter.getTrotterGates(N, 2, onsiteTermsA, neighborTermsA, dt)
            for i in range(int(T / dt)):
                [psiCopy, truncErr] = trotter.trotterSweep(trotterGates, psiCopy, 0, int(len(psiCopy) / 2 - 1))
        psiCopy2 = bops.copyState(psiCopy)
        projectS(psiCopy, int(len(psiCopy)/2 - 1))
        p = bops.getOverlap(psiCopy, psiCopy2)
        sum += p * p
        bops.removeState(psiCopy)
        bops.removeState(psiCopy2)
    avg = sum / NU
    results[e] = avg * (2**int(len(psi)/2) * (2**int(len(psi)/2) + 1)) - 1

plt.plot(etas, [R2] * len(etas))
plt.plot(etas, results)
plt.show()
