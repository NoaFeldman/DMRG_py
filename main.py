import tensornetwork as tn
import numpy as np
import basicOperations as bops
import DMRG as dmrg
import experimantalRandom as exr
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

def noHermitianTest(psi, vs):
    n = len(vs)
    NA = len(vs[0])
    result = 1
    for copy in range(n):
        psiCopy = bops.copyState(psi)
        for alpha in range(NA - 1, -1, -1):
            toEstimate = np.kron(vs[copy][alpha], np.conj(np.reshape(vs[np.mod(copy+1, n)][alpha], [2, 1])))
            hermitianComponent = np.random.randint(2) # choices[copy][alpha]
            if hermitianComponent:
                toMeasure = (toEstimate + np.conj(np.transpose(toEstimate))) / 2
            else:
                toMeasure = (toEstimate - np.conj(np.transpose(toEstimate))) / 2
            psiCopy[alpha] = bops.permute(bops.multiContraction(psiCopy[alpha], tn.Node(toMeasure), '1', '0'), [0, 2, 1])
        result *= bops.getOverlap(psi, psiCopy)
    return result


n=4
for ASize in [8, 6, 4, 2]:
    print('ASize = ' + str(ASize))
    print('n = ' + str(n))

    Sn = bops.getRenyiEntropy(psi, n, ASize - 1)
    print('Sn = ' + str(Sn))
    mySum = 0
    M = 1000
    from datetime import datetime
    for k in range(N - 1, ASize - 1, -1):
        psi = bops.shiftWorkingSite(psi, k, '<<')
    start = datetime.now()
    for m in range(M * 100 * 2**ASize):
        vs = [[np.array([np.random.randint(2) * 2 - 1, np.random.randint(2) * 2 - 1]) \
               for alpha in range(ASize)] for copy in range(n)]
        # mySum += noHermitianTest(psi, vs)
        mySum += exr.singleMeasurement(psi, vs)
        # for choiceCopy in range(2**n):
        #     for choiceAlpha in range(2**ASize):
        #         choices = [[choiceCopy & 2**j > 0 for j in range(n)],
        #                    [choiceAlpha & 2**j > 0 for j in range(ASize)]]
        #         c = noHermitianTest(psi, vs, choices)
                # if np.round(c, 8) > 0:
                #     csum = 0
                #     steps = 10000
                #     for i in range(steps):
                #         csum += exr.singleMeasurement(psi, vs, choices)
                #     cavg = csum / steps
                #     print(csum / steps)
        if m % M == M - 1:
            plt.scatter(m, Sn / (mySum / m))
            # print(mySum / m)
            # print('pn / result = ' + str(Sn / (mySum / m)))
            end = datetime.now()
            # print((end - start).seconds)
    plt.savefig('experimental_n_' + str(n) + '_N_' + str(ASize) + '.png')
    plt.clf()

