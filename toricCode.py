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
currBA = bops.permute(bops.multiContraction(currBA, envOpAB, '45', '01'), [0, 2, 4, 6, 1, 3, 5, 7])

opAB = np.reshape(np.transpose(currAB.tensor, [1, 2, 5, 6, 3, 7, 0, 4]), [16, 16, 16, 16])

openA = tn.Node(np.transpose(np.reshape(np.kron(A.tensor, A.tensor), [d**2, d**2, d**2, d**2, d, d]), [4, 0, 1, 2, 3, 5]))
openB = tn.Node(np.transpose(np.reshape(np.kron(B.tensor, B.tensor), [d**2, d**2, d**2, d**2, d, d]), [4, 0, 1, 2, 3, 5]))

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

upRow = bops.unifyLegs(bops.unifyLegs(bops.unifyLegs(bops.unifyLegs(
    bops.permute(bops.multiContraction(row, tn.Node(currAB.tensor), '12', '04'), [0, 2, 3, 4, 7, 1, 5, 6]), 5, 6), 5, 6), 0, 1), 0, 1)
[C, D, te] = bops.svdTruncation(upRow, [0, 1], [2, 3], '>>', normalize=True)
upRow = bops.multiContraction(D, C, '2', '0')
[cUp, dUp, te] = bops.svdTruncation(upRow, [0, 1], [2, 3], '>>', normalize=True)

GammaC, LambdaC, GammaD, LambdaD = peps.getBMPSRowOps(cUp, tn.Node(np.ones(cUp[2].dimension)), dUp,
                                            tn.Node(np.ones(dUp[2].dimension)), AEnv, BEnv, 50)
cUp = bops.multiContraction(GammaC, LambdaC, '2', '0', isDiag2=True)
dUp = bops.multiContraction(GammaD, LambdaD, '2', '0', isDiag2=True)
upRow = bops.multiContraction(cUp, dUp, '2', '0')
downRow = bops.copyState([upRow])[0]
rightRow = peps.bmpsCols(upRow, downRow, AEnv, BEnv, 50, option='right', X=upRow)
leftRow = peps.bmpsCols(upRow, downRow, AEnv, BEnv, 50, option='left', X=upRow)

circleU = bops.multiContraction(bops.multiContraction(bops.multiContraction(upRow, upRow, '3', '0'), upRow, '5', '0'), upRow, '70', '03')
circle = bops.multiContraction(bops.multiContraction(bops.multiContraction(upRow, rightRow, '3', '0'), upRow, '5', '0'), leftRow, '70', '03')
ABNet = bops.permute(
        bops.multiContraction(bops.multiContraction(openB, openA, '2', '4'), bops.multiContraction(openA, openB, '2', '4'), '28', '16',
                              cleanOr1=True, cleanOr2=True),
        [1, 5, 6, 13, 14, 9, 10, 2, 0, 4, 8, 12, 3, 7, 11, 15])
dm = bops.multiContraction(circle, ABNet, '01234567', '01234567')
ordered = np.round(np.reshape(dm.tensor, [16, 16]), 14)
b = 1

# Make a 4*4 net of applied operators for applyLocalOperators below.
def makeNet(openA, openB, ops):
    openSites = [openB, openA, openA, openB]
    sites = []
    for i in range(4):
        site = tn.Node(np.trace(bops.multiContraction(ops[i], openSites[i], '1', '0').tensor, axis1=0, axis2=5))
        sites.append(site)
    net = bops.permute(bops.multiContraction(bops.multiContraction(sites[0], sites[1], '0', '3'),
                                 bops.multiContraction(sites[2], sites[3], '0', '3'), '15', '03', cleanOr1=True, cleanOr2=True),
                        [0, 2, 3, 6, 7, 4, 5, 1])
    bops.removeState(sites)
    return net


# Order of ops is square by square, as the DM in the commented out section below.
# note that op is the operator acting on the traced DM, eg UPU^\dagger
def applyLocalOperators(upRow, downRow, leftRow, rightRow, openA, openB, ops, l):
    left = leftRow
    for i in range(l):
        leftC = bops.multiContraction(bops.multiContraction(downRow, left, '3', '0'), upRow, '5', '0')
        net = makeNet(openA, openB, ops[i * 4: (i + 1) * 4])
        left = bops.permute(bops.multiContraction(leftC, net, '123456', '456701'), [0, 3, 2, 1])
    return bops.multiContraction(left, rightRow, '0123', '3210').tensor * 1

l = 1
norm = applyLocalOperators(upRow, downRow, leftRow, rightRow, openA, openB, [tn.Node(np.eye(d)) for i in range(l*4)], l)
norm = applyLocalOperators(upRow, downRow, bops.multNode(leftRow, 1 / norm), rightRow, openA, openB, [tn.Node(np.eye(d)) for i in range(l*4)], l)

b = 1




# leftC = bops.multiContraction(bops.multiContraction(downRow, leftRow, '3', '0'), upRow, '5', '0')
# left = bops.permute(bops.multiContraction(leftC, ABNet, '123456', '456701'), [0, 3, 2, 1, 4, 5, 6, 7, 8, 9, 10, 11])
# rightC = bops.multiContraction(bops.multiContraction(upRow, rightRow, '3', '0'), downRow, '5', '0')
# right = bops.permute(bops.multiContraction(rightC, ABNet, '123456', '012345'), [0, 3, 2, 1, 4, 5, 6, 7, 8, 9, 10, 11])
# mid = bops.permute(bops.multiContraction(bops.multiContraction(upRow, ABNet, '12', '01'), downRow, '45', '12'),
#                    [0, 5, 4, 15, 1, 2, 3, 14, 6, 7, 8, 9, 10, 11, 12, 13])
# right = bops.permute(bops.multiContraction(mid, right, '4567', '0123'),
#                      [0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 8, 9, 10, 11, 16, 17, 18, 19])
# dm = bops.permute(bops.multiContraction(left, right, '0123', '3210'), [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15])
# ordered = np.round(np.reshape(dm.tensor, [2**8, 2**8]), 14)
# b = 1

