import numpy as np
import basicOperations as bops
import tensornetwork as tn
import PEPS as peps
import scipy
import math

d = 2

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

# ATensor = np.zeros((2, 2, 2, 2, d))
# BTensor = np.zeros((2, 2, 2, 2, d))
# for i in range(2):
#     for j in range(2):
#         ATensor[i, i, j, j, 0] = 1
#         ATensor[i, i, j, j, 1] = 1 * (int(i == j) * 2 - 1)
#         BTensor[i, j, j, i, 0] = 1
#         BTensor[i, j, j, i, 1] = 1 * (int(i == j) * 2 - 1)
# A = tn.Node(ATensor)
# B = tn.Node(BTensor)


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
    [C, D, te] = bops.svdTruncation(curr, [curr[0], curr[1], curr[2], curr[3]],
                                    [curr[4], curr[5], curr[6], curr[7]], '><', normalize=True)
    curr = bops.permute(bops.multiContraction(D, C, '23', '12'), [1, 3, 0, 5, 2, 4])
    curr = bops.permute(bops.multiContraction(curr, envOpAB, '45', '01'), [0, 2, 4, 6, 1, 3, 5, 7])

currAB = curr
[C, D, te] = bops.svdTruncation(curr, [curr[0], curr[1], curr[2], curr[3]],
                                    [curr[4], curr[5], curr[6], curr[7]], '><', normalize=True)
currBA = bops.permute(bops.multiContraction(D, C, '23', '12'), [1, 3, 0, 5, 2, 4])

opAB = np.reshape(np.transpose(currAB.tensor, [1, 2, 5, 6, 3, 7, 0, 4]), [16, 16, 16, 16])

openA = tn.Node(np.transpose(np.reshape(np.kron(A.tensor, A.tensor), [d**2, d**2, d**2, d**2, d, d]), [4, 0, 1, 2, 3, 5]))
openB = tn.Node(np.transpose(np.reshape(np.kron(B.tensor, B.tensor), [d**2, d**2, d**2, d**2, d, d]), [4, 0, 1, 2, 3, 5]))
ABNet = bops.permute(bops.multiContraction(bops.multiContraction(openB, openA, '2', '4'),
                                           bops.multiContraction(openA, openB, '2', '4'), '28', '16', cleanOriginal2=True,
                                           cleanOriginal1=True),
                     [2, 10, 9, 13, 14, 5, 6, 1, 8, 12, 0, 4, 11, 15, 3, 7])
# ab = np.reshape(ABNet.tensor, [4, 4, 4, 4, 4, 4, 4, 4, 16, 16])
# res = []
# for i in range(4):
#     for j in range(4):
#         for k in range(4):
#             for l in range(4):
#                 for m in range(4):
#                     for n in range(4):
#                         for o in range(4):
#                             for p in range(4):
#                                 for r in range(16):
#                                     if ab[i, j, k, l, m, n, o, p, r, r] > 0:
#                                         res.append([i, j, k, l, m, n, o, p, r])


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


# rowTensor = np.zeros((4, 4, 4, 4), dtype=complex)
# for i in range(4):
#     for j in range(4):
#         left = (i % 2) * 3
#         right = int(np.floor(i / 2)) * 3
#         rowTensor[i, left, right, j] = 1
row = tn.Node(rowTensor)

opRow = bops.unifyLegs(bops.unifyLegs(bops.unifyLegs(bops.unifyLegs(
    bops.permute(bops.multiContraction(row, curr, '12', '04'), [0, 2, 3, 4, 7, 1, 5, 6]), 5, 6), 5, 6), 0, 1), 0, 1)
[C, D, te] = bops.svdTruncation(opRow, [opRow[0], opRow[1]], [opRow[2], opRow[3]], '>>', normalize=True)
opRow = bops.multiContraction(D, C, '2', '0')
[C, D, te] = bops.svdTruncation(opRow, [opRow[0], opRow[1]], [opRow[2], opRow[3]], '>>', normalize=True)
opRow = bops.multiContraction(D, C, '2', '0')

# import entanglementEstimation as ee
# ee.getMeasurement(A, B, C, D, 4, 4)

L = bops.multiContraction(opRow, opRow, '3', '0')
circle = bops.multiContraction(L, L, '05', '50')
dm = bops.multiContraction(circle, ABNet, '01234567', '01234567')
ordered = np.round(np.reshape(dm.tensor, [16,  16]), 13)
ordered = ordered / np.trace(ordered)

"""A Random matrix distributed with Haar measure"""
def haar_measure(n):
    z = (np.random.randn(n, n) + 1j * np.random.randn(n, n)) / np.sqrt(2.0)
    q,r = scipy.linalg.qr(z)
    d = np.diagonal(r)
    ph = d / np.absolute(d)
    q = np.multiply(q,ph,q)
    return q

def randomMeas(rho):
    r = np.random.uniform()
    accum = 0
    for i in range(len(rho)):
        if r > accum and r < accum + rho[i, i]:
            return i
        accum += rho[i, i]
    b = 1

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
    env0 = bops.permute(bops.multiContraction(bops.multiContraction(bops.multiContraction(
        circle, A1, '34', '01'), B3, '634', '012'), A2, '530', '123'), [1, 2, 3, 0])
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