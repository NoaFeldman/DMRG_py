import numpy as np
import basicOperations as bops
import tensornetwork as tn
import PEPS as peps

beta = 1
d = 2

baseTensor = np.zeros((d, d, d), dtype=complex)
for i in range(d):
    baseTensor[i, 0, i] = 1 #np.sqrt(np.cosh(beta))
for i in range(d):
    baseTensor[i, 1, i] = 0.5 * (1 - 2 * i) #np.sqrt(np.sinh(beta)) * (1 - 2 * i) # sigma_z

base = tn.Node(baseTensor)

A = bops.multiContraction(
    bops.multiContraction(bops.multiContraction(base, base, '2', '0'), base, '3', '0', cleanOr1=True), base, '4', '0',
    cleanOr1=True)
AEnv = tn.Node(np.trace(A.get_tensor(), axis1=0, axis2=5))

GammaATensor = np.zeros((d, d, d), dtype=complex)
GammaATensor[1, 1, 1] = 1
GammaATensor[0, 0, 0] = 1
GammaBTensor = np.zeros((d, d, d), dtype=complex)
GammaBTensor[1, 1, 1] = 1
GammaBTensor[1, 1, 0] = -1
GammaBTensor[0, 0, 1] = 1
GammaBTensor[0, 0, 0] = 1
LambdaTensor = np.ones(d, dtype=complex)
cUp, dUp, cDown, dDown = peps.getBMPSRowOps(
    tn.Node(GammaATensor), tn.Node(LambdaTensor), tn.Node(GammaBTensor), tn.Node(LambdaTensor), AEnv, AEnv, 100)
xRight = peps.bmpsSides(cUp, dUp, cUp, dUp, AEnv, AEnv, 100, option='right')
xLeft = peps.bmpsSides(cUp, dUp, cUp, dUp, AEnv, AEnv, 100, option='left')

pair = bops.permute(bops.multiContraction(A, A, '2', '4'), [1, 6, 3, 7, 2, 8, 0, 5, 4, 9])
upRow = bops.multiContraction(cUp, dUp, '2', '0')
downRow = bops.multiContraction(cDown, dDown, '2', '0')


def estimateOp(xRight, xLeft, upRow, downRow, A, ops):
    N = len(ops)
    curr = xLeft
    for i in range(int(N / 2)):
        closedA = tn.Node(np.trace(bops.multiContraction(ops[i * 2], A, '1', '0').tensor, axis1=0, axis2=5))
        closedB = tn.Node(np.trace(bops.multiContraction(ops[i * 2 + 1], A, '1', '0').tensor, axis1=0, axis2=5))
        closed = bops.permute(bops.multiContraction(closedA, closedB, '1', '3'), [0, 3, 2, 4, 1, 5])
        curr = bops.multiContraction(bops.multiContraction(bops.multiContraction(
            curr, upRow, '0', '0'), closed, '023', '201', cleanOr1=True), downRow, '034', '012', cleanOr1=True)
    return bops.multiContraction(curr, xRight, '012', '012').tensor


l = 14
t = estimateOp(xRight, xLeft, upRow, upRow, A, [tn.Node(np.eye(2)) for i in range(l)])
xLeft = bops.multNode(xLeft, 1 / t)
t = estimateOp(xRight, xLeft, upRow, upRow, A, [tn.Node(np.eye(2)) for i in range(l)])
projector0 = np.zeros((2, 2))
projector0[0, 0] = 1
proj0 = tn.Node(projector0)
projector1 = np.zeros((2, 2))
projector1[1, 1] = 1
proj1 = tn.Node(projector1)
tr = 0
for i in range(2**l):
    ops = []
    for b in range(l):
        if i & 2**b == 0:
            ops.append(proj0)
        else:
            ops.append(proj1)
    p = estimateOp(xRight, xLeft, upRow, upRow, A, ops)
    tr += p
b = 1

