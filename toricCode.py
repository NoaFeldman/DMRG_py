import numpy as np
import basicOperations as bops
import tensornetwork as tn
import PEPS as peps
import randomUs as ru
import pickle

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


def toEnvOperator(op):
    result = bops.unifyLegs(bops.unifyLegs(bops.unifyLegs(bops.unifyLegs(
        bops.permute(op, [0, 4, 1, 5, 2, 6, 3, 7]), 6, 7), 4, 5), 2, 3), 0, 1)
    tn.remove_node(op)
    return result
AEnv = toEnvOperator(bops.multiContraction(A, A, '4', '4*'))
BEnv = toEnvOperator(bops.multiContraction(B, B, '4', '4*'))
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
#
# curr = bops.permute(bops.multiContraction(envOpBA, envOpAB, '45', '01'), [0, 2, 4, 6, 1, 3, 5, 7])
#
# for i in range(50):
#     [C, D, te] = bops.svdTruncation(curr, [0, 1, 2, 3], [4, 5, 6, 7], '>>', normalize=True)
#     curr = bops.permute(bops.multiContraction(D, C, '23', '12'), [1, 3, 0, 5, 2, 4])
#     curr = bops.permute(bops.multiContraction(curr, envOpAB, '45', '01'), [0, 2, 4, 6, 1, 3, 5, 7])
#
# currAB = curr
# [C, D, te] = bops.svdTruncation(curr, [0, 1, 2, 3], [4, 5, 6, 7], '>>', normalize=True)
# currBA = bops.permute(bops.multiContraction(D, C, '23', '12'), [1, 3, 0, 5, 2, 4])
# currBA = bops.permute(bops.multiContraction(currBA, envOpAB, '45', '01'), [0, 2, 4, 6, 1, 3, 5, 7])
#
# opAB = np.reshape(np.transpose(currAB.tensor, [1, 2, 5, 6, 3, 7, 0, 4]), [16, 16, 16, 16])
#
# openA = tn.Node(np.transpose(np.reshape(np.kron(A.tensor, A.tensor), [d**2, d**2, d**2, d**2, d, d]), [4, 0, 1, 2, 3, 5]))
# openB = tn.Node(np.transpose(np.reshape(np.kron(B.tensor, B.tensor), [d**2, d**2, d**2, d**2, d, d]), [4, 0, 1, 2, 3, 5]))
#
# rowTensor = np.zeros((11, 4, 4, 11), dtype=complex)
# rowTensor[0, 0, 0, 0] = 1
# rowTensor[1, 0, 0, 2] = 1
# rowTensor[2, 0, 0, 3] = 1
# rowTensor[3, 0, 3, 4] = 1
# rowTensor[4, 3, 0, 1] = 1
# rowTensor[5, 0, 0, 6] = 1
# rowTensor[6, 0, 3, 7] = 1
# rowTensor[7, 3, 0, 8] = 1
# rowTensor[8, 0, 0, 5] = 1
# row = tn.Node(rowTensor)
#
# upRow = bops.unifyLegs(bops.unifyLegs(bops.unifyLegs(bops.unifyLegs(
#     bops.permute(bops.multiContraction(row, tn.Node(currAB.tensor), '12', '04'), [0, 2, 3, 4, 7, 1, 5, 6]), 5, 6), 5, 6), 0, 1), 0, 1)
# [C, D, te] = bops.svdTruncation(upRow, [0, 1], [2, 3], '>>', normalize=True)
# upRow = bops.multiContraction(D, C, '2', '0')
# [cUp, dUp, te] = bops.svdTruncation(upRow, [0, 1], [2, 3], '>>', normalize=True)
#
# GammaC, LambdaC, GammaD, LambdaD = peps.getBMPSRowOps(cUp, tn.Node(np.ones(cUp[2].dimension)), dUp,
#                                             tn.Node(np.ones(dUp[2].dimension)), AEnv, BEnv, 50)
# cUp = bops.multiContraction(GammaC, LambdaC, '2', '0', isDiag2=True)
# dUp = bops.multiContraction(GammaD, LambdaD, '2', '0', isDiag2=True)
# upRow = bops.multiContraction(cUp, dUp, '2', '0')
# downRow = bops.copyState([upRow])[0]
# rightRow = peps.bmpsCols(upRow, downRow, AEnv, BEnv, 50, option='right', X=upRow)
# leftRow = peps.bmpsCols(upRow, downRow, AEnv, BEnv, 50, option='left', X=upRow)
#
# circle = bops.multiContraction(bops.multiContraction(bops.multiContraction(upRow, rightRow, '3', '0'), upRow, '5', '0'), leftRow, '70', '03')
# ABNet = bops.permute(
#         bops.multiContraction(bops.multiContraction(openB, openA, '2', '4'), bops.multiContraction(openA, openB, '2', '4'), '28', '16',
#                               cleanOr1=True, cleanOr2=True),
#         [1, 5, 6, 13, 14, 9, 10, 2, 0, 4, 8, 12, 3, 7, 11, 15])
# dm = bops.multiContraction(circle, ABNet, '01234567', '01234567')
# ordered = np.round(np.reshape(dm.tensor, [16, 16]), 14)
# ordered /= np.trace(ordered)
# b = 1
#
# with open('toricBoundaries', 'wb') as f:
#     pickle.dump([upRow, downRow, leftRow, rightRow, openA, openB], f)
#


def applyOpTosite(site, op):
    return toEnvOperator(bops.multiContraction(bops.multiContraction(site, op, '4', '1'), site, '4', '4*'))


def applyLocalOperators(cUp, dUp, cDown, dDown, leftRow, rightRow, A, B, l, ops):
    left = leftRow
    for i in range(l):
        left = bops.multiContraction(left, cUp, '3', '0', cleanOr1=True)
        lu = applyOpTosite(B, ops[i * 4])
        ld = applyOpTosite(A, ops[i * 4 + 2])
        left = bops.multiContraction(left, lu, '23', '30', cleanOr1=True)
        left = bops.multiContraction(left, ld, '14', '30', cleanOr1=True)
        left = bops.permute(bops.multiContraction(left, dDown, '04', '21', cleanOr1=True), [3, 2, 1, 0])

        left = bops.multiContraction(left, dUp, '3', '0', cleanOr1=True)
        ru = applyOpTosite(A, ops[i * 4 + 1])
        rd = applyOpTosite(B, ops[i * 4 + 3])
        left = bops.multiContraction(left, ru, '23', '30', cleanOr1=True)
        left = bops.multiContraction(left, rd, '14', '30', cleanOr1=True)
        left = bops.permute(bops.multiContraction(left, cDown, '04', '21', cleanOr1=True), [3, 2, 1, 0])

        bops.removeState([lu, ld, rd, ru])

    return bops.multiContraction(left, rightRow, '0123', '3210').tensor * 1


with open('toricBoundaries', 'rb') as f:
    [upRow, downRow, leftRow, rightRow, openA, openB] = pickle.load(f)

upRow = tn.Node(upRow)
downRow = tn.Node(downRow)
leftRow = tn.Node(leftRow)
rightRow = tn.Node(rightRow)
openA = tn.Node(openA)
openB = tn.Node(openB)

M = 1000
avg = 0
l = 3
# [cUp, dUp, te] = bops.svdTruncation(upRow, [0, 1], [2, 3], '>>')
# [cDown, dDown, te] = bops.svdTruncation(downRow, [0, 1], [2, 3], '>>')
#
# norm = applyLocalOperators(cUp, dUp, cDown, dDown, leftRow, rightRow, A, B, l,
#                                [tn.Node(np.eye(d)) for i in range(l * 4)])
# leftRow = bops.multNode(leftRow, 1 / norm)

circle = bops.multiContraction(bops.multiContraction(bops.multiContraction(upRow, rightRow, '3', '0'), upRow, '5', '0'), leftRow, '70', '03')
ABNet = bops.permute(
        bops.multiContraction(bops.multiContraction(openB, openA, '2', '4'), bops.multiContraction(openA, openB, '2', '4'), '28', '16',
                              cleanOr1=True, cleanOr2=True),
        [1, 5, 6, 13, 14, 9, 10, 2, 0, 4, 8, 12, 3, 7, 11, 15])
ab = np.reshape(ABNet.tensor, [4**2, 4**2, 4**2, 4**2, 2**4, 2**4])
for i0 in range(4**2):
    for i1 in range(4**2):
        for i2 in range(4**2):
            for i3 in range(4**2):
                for j in range(8):
                    expected = np.zeros((16, 16))
                    expected[j, j] = 1
                    expected[j, j^15] = 1
                    expected[j^15, j] = 1
                    expected[j^15, j^15] = 1
                if np.all(expected > 0, ab[i0, i1, i2, i3] > 0):
                    b = 1
dm = bops.multiContraction(circle, ABNet, '01234567', '01234567')
ordered = np.round(np.reshape(dm.tensor, [16, 16]), 14)
ordered /= np.trace(ordered)
b = 1
left = bops.multiContraction(circle, ABNet, '123456', '456701')
circle = bops.multiContraction(bops.multiContraction(downRow, left, '3', '0'), upRow, '3', '0')
left = bops.permute(bops.multiContraction(circle, ABNet, [1, 2, 4, 3, 13, 14], '456701'),
                    [0, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17, 18, 19])
dm = bops.permute(bops.multiContraction(left, rightRow, '1230', '0123'),
                  [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15])
ordered = np.reshape(dm.tensor, [2**8, 2**8]) / np.trace(np.reshape(dm.tensor, [2**8, 2**8]))
b = 1


# circle = bops.multiContraction(bops.multiContraction(bops.multiContraction(upRow, rightRow, '3', '0'), upRow, '5', '0'), leftRow, '70', '03')
# ABNet = bops.permute(
#         bops.multiContraction(bops.multiContraction(openB, openA, '2', '4'), bops.multiContraction(openA, openB, '2', '4'), '28', '16',
#                               cleanOr1=True, cleanOr2=True),
#         [1, 5, 6, 13, 14, 9, 10, 2, 0, 4, 8, 12, 3, 7, 11, 15])
# dm = bops.multiContraction(circle, ABNet, '01234567', '01234567')
# ordered = np.round(np.reshape(dm.tensor, [16, 16]), 14)
# ordered /= np.trace(ordered)
#
# proj0 = np.zeros((2, 2))
# proj0[0, 0] = 1
# proj1 = np.zeros((2, 2))
# proj1[1, 1] = 1
# flip0to1 = np.zeros((2, 2))
# flip0to1[1, 0] = 1
# flip1to0 = np.zeros((2, 2))
# flip1to0[0, 1] = 1
#


# M = 1000
# chi = 300
# for l in range(1, 2):
#     norm = applyLocalOperators(upRow, downRow, leftRow, rightRow, openA, openB, l,
#                                [tn.Node(np.eye(d)) for i in range(l * 4)])
#     leftRow = bops.multNode(leftRow, 1 / norm)
#     ru.localUnitariesFull(l * 4, M, applyLocalOperators, [upRow, downRow, leftRow, rightRow, openA, openB, l], 'toric_local_full')
