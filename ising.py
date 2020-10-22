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
    tn.Node(GammaATensor), tn.Node(LambdaTensor), tn.Node(GammaBTensor), tn.Node(LambdaTensor), AEnv, AEnv, 50)
dm = peps.bmpsDensityMatrix(cUp, dUp, cDown, dDown, AEnv, AEnv, A, A, 50)
ordered = np.round(np.reshape(dm.tensor, [16,  16]), 13)
p2 = sum(np.diag(ordered)**2)

fourierTensor = np.zeros((4, 4), dtype=complex)
fourierTensor[0, 0] = 1
fourierTensor[0, 3] = 1
fourierTensor[1, 1] = 1 / np.sqrt(2)
fourierTensor[1, 2] = 1 / np.sqrt(2)
fourierTensor[2, 1] = -1 / np.sqrt(2)
fourierTensor[2, 2] = 1 / np.sqrt(2)
fourier = tn.Node(fourierTensor)
doubleA = tn.Node(np.kron(A.tensor, A.tensor))
fA = bops.multiContraction(bops.multiContraction(fourier, doubleA, '1', '0'), fourier, '5', '1*', cleanOr1=True)
fAEnv = tn.Node(np.trace(fA.get_tensor(), axis1=0, axis2=5))
cUp, dUp, cDown, dDown = peps.getBMPSRowOps(
    tn.Node(np.ones((2, 4, 2))), tn.Node(np.ones((2))), tn.Node(np.ones((2, 4, 2))), tn.Node(np.ones((2))), fAEnv, fAEnv, 100)
dm = peps.bmpsDensityMatrix(cUp, dUp, cDown, dDown, fAEnv, fAEnv, fA, fA, 100)
b=1