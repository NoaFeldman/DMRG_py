import numpy as np
import basicOperations as bops
import tensornetwork as tn
import PEPS as peps
from matplotlib import pyplot as plt


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




