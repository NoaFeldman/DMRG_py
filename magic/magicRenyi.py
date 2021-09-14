import tensornetwork as tn
import numpy as np
import magic.basicDefs as basicdefs
from typing import List
import basicOperations as bops

# Vectorize all matrices. trPs, trPDaggers are built such that applying them to the vectorized DM we get the vector
# (tr(\rho), tr(X\rho), tr(Y\rho), tr(Z\rho)), or its eqiovalent in general d.
# the state returned is a state with each site representing a tensor product of 4 copies of the state,
# two regular and two conjugated.
def get4PsiVectorized(psi: List[tn.Node], d: int):
    paulis = basicdefs.getPauliMatrices(d)
    tracePs = np.zeros((d**4, d**4), dtype=complex)
    for i in range(len(paulis)):
        p = paulis[i]
        traceP = p.transpose().reshape([d**2])
        tracePs[i, :] = np.kron(traceP, traceP)
    psi4 = []
    for i in range(len(psi)):
        temp = psi[i].tensor.reshape([1] + list(psi[i].shape))
        chiL = psi[i].edges[0].dimension
        chiR = psi[i].edges[2].dimension
        unifier = np.ones((1, 1, 1, 1))
        tensorProd = np.tensordot(np.tensordot(np.tensordot(np.tensordot(unifier, temp, axes=(3, 0)), np.conj(temp), axes=(2, 0)),
                                  temp, axes=(1, 0)), np.conj(temp), axes=(0, 0))
        tensorProd = tensorProd.transpose([0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]).reshape([chiL**4, d**4, chiR**4])
        psi4.append(tn.Node(tensorProd))
    return psi4, tn.Node(tracePs / d)


def getSecondRenyi(psi, d):
    n = len(psi)
    psi4, trPs = get4PsiVectorized(psi, d)
    for i in range(n):
        bops.applySingleSiteOp(psi4, trPs, i)
    renyiSum = bops.getOverlap(psi4, psi4)
    return -np.log(renyiSum) / np.log(d) - n
