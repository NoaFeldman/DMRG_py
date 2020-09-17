import numpy as np
import basicOperations as bops
import tensornetwork as tn
import PEPS as peps

beta = 1
d = 2

baseTensor = np.zeros((d, d, d), dtype=complex)
for i in range(d):
    baseTensor[i, 0, i] = np.sqrt(np.cosh(beta))
for i in range(d):
    baseTensor[i, 1, i] = np.sqrt(np.sinh(beta)) * (1 - 2 * i) # sigma_z

base = tn.Node(baseTensor)

A = bops.multiContraction(bops.multiContraction(bops.multiContraction(
    base, base, '2', '0'), base, '3', '0', cleanOriginal1=True), base, '4', '0', cleanOriginal1=True)
AEnv = tn.Node(np.trace(A.get_tensor(), axis1=0, axis2=5))

GammaTensor = np.zeros((d, d, d), dtype=complex)
GammaTensor[0, 0, 1] = 1
GammaTensor[1, 1, 0] = 1
LambdaTensor = np.eye(2, dtype=complex)
peps.getBMPSRowOps(tn.Node(GammaTensor), tn.Node(LambdaTensor), tn.Node(GammaTensor), tn.Node(LambdaTensor), AEnv, AEnv, 50)
