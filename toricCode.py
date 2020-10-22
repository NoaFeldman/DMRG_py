import numpy as np
import basicOperations as bops
import tensornetwork as tn
import PEPS as peps
import scipy
import math
from matplotlib import pyplot as plt


d = 2


def expectedDensityMatrix(height, width=2):
    if width != 2:
        # TODO
        return
    rho = np.zeros((d**(height * width), d**(height * width)))
    for i in range(d**(height * width)):
        b = 1
        for j in range(d**(height * width)):
            xors = i ^ j
            counter = 0
            # Look for pairs of reversed sites and count them
            while xors > 0:
                if xors & 3 == 3:
                    counter += 1
                elif xors & 3 == 1 or xors & 3 == 2:
                    counter = -1
                    xors = 0
                xors = xors >> 2
            if counter % 2 == 0:
                rho[i, j] = 1
    rho = rho / np.trace(rho)
    return rho


# Toric code model matrices - figure 30 here https://arxiv.org/pdf/1306.2164.pdf
baseTensor = np.zeros((d, d, d, d), dtype=complex)
baseTensor[0, 0, 0, 0] = 1 / 2**0.25
baseTensor[1, 0, 0, 1] = 1 / 2**0.25
baseTensor[0, 1, 1, 1] = 1 / 2**0.25
baseTensor[1, 1, 1, 0] = 1 / 2**0.25
base = tn.Node(baseTensor)
ABTensor = bops.multiContraction(base, base, '3', '0').tensor[0]
A = tn.Node(ABTensor)
B = tn.Node(np.transpose(ABTensor, [1, 2, 3, 0, 4]))


AEnv = bops.permute(bops.multiContraction(A, A, '4', '4*'), [0, 4, 1, 5, 2, 6, 3, 7])
AEnv = bops.unifyLegs(bops.unifyLegs(bops.unifyLegs(bops.unifyLegs(AEnv, 6, 7), 4, 5), 2, 3), 0, 1)
BEnv = bops.permute(bops.multiContraction(B, B, '4', '4*'), [0, 4, 1, 5, 2, 6, 3, 7])
BEnv = bops.unifyLegs(bops.unifyLegs(bops.unifyLegs(bops.unifyLegs(BEnv, 6, 7), 4, 5), 2, 3), 0, 1)
chi = 32
nonPhysicalLegs = 1
GammaTensor = np.ones((nonPhysicalLegs, d**2, nonPhysicalLegs), dtype=complex)
GammaC = tn.Node(GammaTensor, name='GammaC', backend=None)
LambdaC = tn.Node(np.eye(nonPhysicalLegs) / np.sqrt(nonPhysicalLegs), backend=None)
GammaD = tn.Node(GammaTensor, name='GammaD', backend=None)
LambdaD = tn.Node(np.eye(nonPhysicalLegs) / np.sqrt(nonPhysicalLegs), backend=None)

steps = 50

envOpAB = bops.permute(bops.multiContraction(AEnv, BEnv, '1', '3'), [0, 3, 2, 4, 1, 5])
envOpBA = bops.permute(bops.multiContraction(BEnv, AEnv, '1', '3'), [0, 3, 2, 4, 1, 5])

curr = bops.permute(bops.multiContraction(envOpBA, envOpAB, '45', '01'), [0, 2, 4, 6, 1, 3, 5, 7])

for i in range(50):
    [C, D, te] = bops.svdTruncation(curr, [0, 1, 2, 3], [4, 5, 6, 7], '>>', normalize=True)
    curr = bops.permute(bops.multiContraction(D, C, '23', '12'), [1, 3, 0, 5, 2, 4])
    curr = bops.permute(bops.multiContraction(curr, envOpAB, '45', '01'), [0, 2, 4, 6, 1, 3, 5, 7])

currAB = curr
[C, D, te] = bops.svdTruncation(curr, [0, 1, 2, 3], [4, 5, 6, 7], '>>', normalize=True)
currBA = bops.permute(bops.multiContraction(D, C, '23', '12'), [1, 3, 0, 5, 2, 4])

opAB = np.reshape(np.transpose(currAB.tensor, [1, 2, 5, 6, 3, 7, 0, 4]), [16, 16, 16, 16])

openA = tn.Node(np.transpose(np.reshape(np.kron(A.tensor, A.tensor), [d**2, d**2, d**2, d**2, d, d]), [4, 0, 1, 2, 3, 5]))
openB = tn.Node(np.transpose(np.reshape(np.kron(B.tensor, B.tensor), [d**2, d**2, d**2, d**2, d, d]), [4, 0, 1, 2, 3, 5]))
ABNet = bops.permute(
    bops.multiContraction(bops.multiContraction(openB, openA, '2', '4'), bops.multiContraction(openA, openB, '2', '4'),
                          '28', '16', cleanOr1=True, cleanOr2=True),
    [2, 10, 9, 13, 14, 5, 6, 1, 8, 12, 0, 4, 11, 15, 3, 7])

rowTensor = np.zeros((11, 4, 4, 11), dtype=complex)
rowTensor[0, 0, 0, 0] = 1
rowTensor[1, 0, 0, 2] = 1
rowTensor[2, 0, 0, 3] = 1
rowTensor[3, 0, 3, 4] = 1
rowTensor[4, 3, 0, 1] = 1
rowTensor[5, 0, 0, 6] = 1
rowTensor[6, 0, 3, 7] = 1
rowTensor[7, 3, 0, 8] = 1
rowTensor[8, 0, 0, 5] = 1
row = tn.Node(rowTensor)

# opRow = bops.unifyLegs(bops.unifyLegs(bops.unifyLegs(bops.unifyLegs(
#     bops.permute(bops.multiContraction(row, curr, '12', '04'), [0, 2, 3, 4, 7, 1, 5, 6]), 5, 6), 5, 6), 0, 1), 0, 1)
# [C, D, te] = bops.svdTruncation(opRow, [0, 1], [2, 3], '>>', normalize=True)
# opRow = bops.multiContraction(D, C, '2', '0')
# [C, D, te] = bops.svdTruncation(opRow, [0, 1], [2, 3], '>>', normalize=True)
# opRow = bops.multiContraction(D, C, '2', '0')
#
# L = bops.multiContraction(opRow, opRow, '3', '0')
# circle = bops.multiContraction(L, L, '05', '50')
# dm = bops.multiContraction(circle, ABNet, '01234567', '01234567')
# ordered = np.round(np.reshape(dm.tensor, [16,  16]), 13)
# ordered = ordered / np.trace(ordered)
# b = 1


"""A Random matrix distributed with Haar measure"""
def haar_measure(n):
    z = (np.random.randn(n, n) + 1j * np.random.randn(n, n)) / np.sqrt(2.0)
    q,r = scipy.linalg.qr(z)
    d = np.diagonal(r)
    ph = d / np.absolute(d)
    q = np.multiply(q,ph,q)
    return q


# create a global unitary from 2 layers of nearest neighbor unitaries
def globalUnitary(N, numberOfLayers=2):
    U = np.eye(d**N)
    for i in range(numberOfLayers):
        u01 = np.kron(haar_measure(d**2), np.eye(d**2, dtype=complex))
        u02 = np.reshape(
            np.transpose(np.reshape(np.kron(haar_measure(d ** 2), np.eye(d ** 2, dtype=complex)), [d] * 2 * N),
                         [0, 2, 1, 3, 4, 6, 5, 7]), [d ** N, d ** N])
        u23 = np.kron(np.eye(d**2, dtype=complex), haar_measure(d**2))
        u13 = np.reshape(np.transpose(np.reshape(np.kron(haar_measure(d**2), np.eye(d**2, dtype=complex)), [d] * 2 * N),
                                      [2, 0, 3, 1, 6, 4, 7, 5]), [d**N, d**N])
        U = np.matmul(U, np.matmul(u01, np.matmul(u02, np.matmul(u23, u13))))
    return U


# def haar_measure_test(n, tests):
#     M = 10000
#     testResults = [0] * len(tests)
#     for m in range(M):
#         U = haar_measure(n)
#         for t in range(len(tests)):
#             testResult = U[tests[t][0], tests[t][1]] * \
#                              np.conj(U[tests[t][2], tests[t][3]]) * \
#                              U[tests[t][4], tests[t][5]] * \
#                              np.conj(U[tests[t][6], tests[t][7]])
#             testResults[t] += testResult
#     testResults = np.round(np.array(testResults) / M, 16)
#     return testResults
# tests = [
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 15, 0, 15, 0, 15, 0, 15],
#     [0, 0, 0, 0, 15, 15, 15, 15],
#     [0, 15, 0, 15, 15, 0, 15, 0],
#     [0, 0, 15, 15, 15, 15, 0, 0],
#     [0, 0, 15, 0, 15, 15, 0, 15],
#     [0, 0, 0, 15, 15, 15, 15, 0],
#     [0, 0, 0, 15, 15, 15, 15, 15],
#     [2, 4, 2, 7, 5, 7, 5, 4]
#     ]
# n = 16
# testAnswers = [2 / (n**2 - 1) - 2 / (n * (n**2 - 1)),
#                2 / (n**2 - 1) - 2 / (n * (n**2 - 1)),
#                1 / (n**2 - 1),
#                1 / (n**2 - 1),
#                1 / (n**2 - 1),
#                -1 / (n * (n**2 - 1)),
#                -1 / (n * (n**2 - 1)),
#                0,
#                -1 / (n * (n**2 - 1))
#                ]
# testResults = haar_measure_test(n, tests)
# b = 1


# https://journals.aps.org/pra/pdf/10.1103/PhysRevA.99.052323
# This is an easy fix for a small DM, not the way it should really go in PEPS
def localDistance(s, sp):
    return bin(s ^ sp).count("1")

def localUnitariesPurityEstimator(N, probabilities, option='allPairs'):
    if option == 'allPairs':
        return d**N * sum([sum([(-d)**(-localDistance(s, sp)) * probabilities[s] * probabilities[sp] \
                        for sp in range(d**N)]) for s in range(d**N)])
    else:
        s = np.random.randint(low=0, high=len(probabilities))
        sp = np.random.randint(low=0, high=len(probabilities))
        return d ** N * (-d) ** (-localDistance(s, sp)) * probabilities[s] * probabilities[sp]


M = 200000
rho = expectedDensityMatrix(2)
avg = 0
avg2 = 0
avgs = []
avg2s = []
systemSize = 4
spaceSize = d**systemSize
step = 100
for m in range(M):
    U = globalUnitary(4, 2)
    # U = haar_measure(spaceSize)
    uRhoU = np.matmul(U, np.matmul(rho, np.conj(np.transpose(U))))
    avg += uRhoU[0, 0]
    avg2 += uRhoU[0, 0]**2
    if m > 0 and m % step == step - 1:
        avgs.append(avg / m)
        avg2s.append(avg2 / m)
avgs = np.array(avgs)
avg2s = np.array(avg2s)
e5s = avgs * 16
e6s = avg2s * 16 * 17 - e5s**2
plt.plot([100 * i - 1 for i in range(int(M / 100))], np.abs(e6s - sum(np.linalg.eigvalsh(rho)**2)))
b = 1


systemLengths = list(range(2, 5))
estimations = []
for l in systemLengths:
    systemSize = 2 * l
    rho = expectedDensityMatrix(l)
    avg = 0
    res = []
    stepsNum = systemSize * 100000
    for m in range(stepsNum):
        U = np.eye(1, dtype=complex)
        for i in range(systemSize):
            U = np.kron(U, haar_measure(d))
        uRhoU = np.matmul(U, np.matmul(rho, np.conj(np.transpose(U))))
        probabilities = np.diag(uRhoU)
        avg += localUnitariesPurityEstimator(systemSize, probabilities, option='singlePair')
        if m > stepsNum / 2 and m % 100 == 99:
           res.append(avg / m)
    estimations.append(res)
b = 1


def randomMeas(rho):
    r = np.random.uniform()
    accum = 0
    for i in range(len(rho)):
        if r > accum and r < accum + rho[i, i]:
            return i
        accum += rho[i, i]
    b = 1

# https://arxiv.org/pdf/2007.06305.pdf
M = 16 * 10000
ks = [[0] * 4 for m in range(M)]
us = [0] * M

for m in range(M):
    u0 = tn.Node(haar_measure(d))
    openB0 = bops.multiContraction(u0, bops.multiContraction(openB, u0, '5', '1*'), '1', '0')
    u1 = tn.Node(haar_measure(d))
    openA1 = bops.multiContraction(u1, bops.multiContraction(openA, u1, '5', '1*'), '1', '0')
    u2 = tn.Node(haar_measure(d))
    openA2 = bops.multiContraction(u2, bops.multiContraction(openA, u2, '5', '1*'), '1', '0')
    u3 = tn.Node(haar_measure(d))
    openB3 = bops.multiContraction(u3, bops.multiContraction(openB, u3, '5', '1*'), '1', '0')
    env123 = bops.multiContraction(circle, BEnv, '12', '03')
    env23 = bops.multiContraction(env123, AEnv, '126', '013')
    env3 = bops.permute(bops.multiContraction(env23, AEnv, '430', '023'), [2, 0, 1, 3])

    us[m] = [u0.tensor, u1.tensor, u2.tensor, u3.tensor]

    rho3 = bops.multiContraction(env3, openB3, '0123', '1234')
    ks[m][3] = randomMeas(rho3.tensor / np.trace(rho3.tensor))
    if ks[m][3] == 0:
        B3 = tn.Node(openB3.tensor[0, :, :, :, :, 0])
    else:
        B3 = tn.Node(openB3.tensor[1, :, :, :, :, 1])
    env2 = bops.permute(bops.multiContraction(env23, B3, '512', '012'), [2, 3, 1, 0])
    rho2 = bops.multiContraction(env2, openA2, '0123', '1234')
    ks[m][2] = randomMeas(rho2.tensor / np.trace(rho2.tensor))
    if ks[m][2] == 0:
        A2 = tn.Node(openA2.tensor[0, :, :, :, :, 0])
    else:
        A2 = tn.Node(openA2.tensor[1, :, :, :, :, 1])
    env1 = bops.permute(bops.multiContraction(bops.multiContraction(env123, A2, '750', '023'), B3, '235', '123')
                        , [0, 1, 3, 2])
    rho1 = bops.multiContraction(env1, openA1, '0123', '1234')
    ks[m][1] = randomMeas(rho1.tensor / np.trace(rho1.tensor))
    if ks[m][1] == 0:
        A1 = tn.Node(openA1.tensor[0, :, :, :, :, 0])
    else:
        A1 = tn.Node(openA1.tensor[1, :, :, :, :, 1])
    env0 = bops.permute(
        bops.multiContraction(bops.multiContraction(bops.multiContraction(circle, A1, '34', '01'), B3, '634', '012'),
                              A2, '530', '123'), [1, 2, 3, 0])
    rho0 = bops.multiContraction(env0, openB0, '0123', '1234')
    ks[m][0] = randomMeas(rho0.tensor / np.trace(rho0.tensor))
import pickle
with open('ks_M160000', 'wb') as fp:
    pickle.dump(ks, fp)
with open('us_M160000', 'wb') as fp:
    pickle.dump(us, fp)

def singleSiteEstimate(k, u):
    proj = np.zeros((d, d), dtype=complex)
    proj[k, k] = 1
    return 3 * np.matmul(np.conj(np.transpose(u)), np.matmul(proj, u)) - np.eye(d, dtype=complex)


purity = 0
factor = M * (M-1)
for m1 in range(M):
    for m2 in range(M):
        if m2 != m1:
            curr = 1
            for i in range(4):
                curr *= np.trace(np.matmul(singleSiteEstimate(ks[m1][i], us[m1][i]),
                                           singleSiteEstimate(ks[m2][i], us[m2][i])))
            purity += curr / factor

negativity = 0
factor = M * M-1 * M-2
for m1 in range(M):
    for m2 in range(M):
        if m2 != m1:
            for m3 in range(M):
                if m3 != m1 and m3 != m2:
                    curr = 1
                    for i in range(2):
                        curr *= np.trace(np.matmul(np.matmul(singleSiteEstimate(ks[m1][i], us[m1][i]),
                                                   singleSiteEstimate(ks[m2][i], us[m2][i])),
                                                   singleSiteEstimate(ks[m3][i], us[m3][i])))
                    for i in range(2, 4):
                        curr *= np.trace(np.matmul(np.matmul(singleSiteEstimate(ks[m1][i], us[m1][i]),
                                                             singleSiteEstimate(ks[m3][i], us[m3][i])),
                                                   singleSiteEstimate(ks[m2][i], us[m2][i])))
                    negativity += curr / factor

b = 1


cUp, dUp, cDown, dDown = peps.getBMPSRowOps(C, S, D, tn.Node(np.eye(D[2].dimension, dtype=complex)), AEnv, BEnv, 10)
dm = peps.bmpsDensityMatrix(cUp, dUp, cDown, dDown, AEnv, BEnv, openA, openB, 50)