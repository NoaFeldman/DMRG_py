from scipy import linalg
import numpy as np
import basicOperations as bops
import randomMeasurements as rm
import sys

d = 3

sX = np.zeros((d, d), dtype=complex)
sX[0, 1] = 1 / np.sqrt(2)
sX[1, 0] = 1 / np.sqrt(2)
sX[1, 2] = 1 / np.sqrt(2)
sX[2, 1] = 1 / np.sqrt(2)
sZ = np.zeros((d, d), dtype=complex)
sZ[0, 0] = -1
sZ[1, 1] = 0
sZ[2, 2] = 1
sY = 1j * (np.matmul(sZ, sX) - np.matmul(sX, sZ))

# sXsX = np.kron(sX, sX)
# sYsY = np.kron(sY, sY)
# sZsZ = np.kron(sZ, sZ)


UtrBase = linalg.expm(1j * np.pi * sY)


def getDistance(s, sP, N):
    sExplicit = [0] * N
    sPExplicit = [0] * N
    for i in range(N):
        sExplicit[i] = s // d**(N - 1 - i)
        sPExplicit[i] = sP // d**(N - 1 - i)
        s = s % d**(N - 1 - i)
        sP = sP % d**(N - 1 - i)
    return N - sum(np.array(sPExplicit) == np.array(sExplicit))


def rotate(mat, U):
    return np.matmul(U, np.matmul(mat, np.conj(np.transpose(U))))


def dagger(mat):
    return np.conj(np.transpose(mat))

def partiallyTranspose(mat, a2Length):
    res = np.zeros(mat.shape, dtype=complex)
    for i in range(a2Length):
        for j in range(a2Length):
            for k in range(a2Length):
                for l in range(a2Length):
                    res[d * i + j, d * k + l] = mat[d * i + l, d * k + j]
    return res


def traceDistance(rho, sigma):
    return sum([np.sqrt(v) for v in np.linalg.eigvals(np.matmul(np.conj(np.transpose(rho - sigma)), rho - sigma))]) / 2


def getProjector(s, spaceSize):
    res = np.zeros((spaceSize, spaceSize))
    res[s, s] = 1
    return res

singleSiteSwap = np.zeros((d**2, d**2), dtype=complex)
for i in range(d):
    for j in range(d):
        singleSiteSwap[i * d + j, j * d + i] = 1
myO = np.zeros((d**2, d**2), dtype=complex)
for i in range(d):
    for j in range(d):
        myO[i * d + i, j * d + j] = 1


N = 32
A1Start = N // 4
A1End = N // 2
lenA1 = A1End - A1Start
A2Start = A1End
A2End = 3 * N // 4
lenA2 = A2End - A2Start
psi = bops.getStartupState(N, d=d, mode='aklt')
M = int(sys.argv[1])
A1Estimations = [[None for i in range(lenA1)] for m in range(M)]
A2Estimations = [[None for i in range(lenA2)] for m in range(M)]

for m in range(M):
    psiCopy = bops.copyState(psi)
    us = [None for i in range(lenA1 + lenA2)]
    for i in range(A1Start, A2End):
        u = rm.haar_measure(d, 's' + str(i))
        us[i - A1Start] = u.tensor
        bops.applySingleSiteOp(psiCopy, u, i)
    meas = rm.randomMeasurement(psiCopy, A1Start, A2End)
    for i in range(lenA1):
        A1Estimations[m][i] = (d + 1) * rotate(getProjector(meas[i], d), dagger(us[i])) - np.eye(d)
    for i in range(lenA2):
        A2Estimations[m][i] = (d + 1) * rotate(getProjector(meas[lenA1 + i], d), dagger(us[lenA1 + i])) - np.eye(d)

import pickle
attemptNumber = sys.argv[2]
output = open('A1Estimations' + attemptNumber + '.pkl', 'wb')
pickle.dump(A1Estimations, output)
output.close()
output = open('A2Estimations' + attemptNumber + '.pkl', 'wb')
pickle.dump(A2Estimations, output)
output.close()
# pkl_file = open('A1Estimations.pkl', 'rb')
# tst = pickle.load(pkl_file)
# b = 1









