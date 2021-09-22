import tensornetwork as tn
import numpy as np
import magic.basicDefs as basicdefs
from typing import List
import basicOperations as bops
import localVecsMPS as randomVecs
import pickle
import os


# Vectorize all matrices. trPs, trPDaggers are built such that applying them to the vectorized DM we get the vector
# (tr(\rho), tr(X\rho), tr(Y\rho), tr(Z\rho)), or its eqiovalent in general d.
# the state returned is a state with each site representing a tensor product of 4 copies of the state,
# two regular and two conjugated.
def get4PsiVectorized(psi: List[tn.Node], d: int):
    paulis = basicdefs.getPauliMatrices(d)
    tracePs = np.zeros((d**2, d**4), dtype=complex)
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


def getPsiPPsiMatrix(psi: List[tn.Node], d: int):
    psi2, tracePs = get2PsiVectorized(psi, d)
    splitter = np.zeros((d**2, d**2, d**2))
    for j in range(d**2):
        splitter[j, j, j] = 1
    splitter = tn.Node(splitter)
    for i in range(len(psi)):
        bops.applySingleSiteOp(psi2, tracePs, i)
        psi2[i] = bops.multiContraction(psi2[i], splitter, '1', '0', cleanOr1=True)
    return psi2


def getSecondRenyi(psi, d):
    n = len(psi)
    psi4, trPs = get4PsiVectorized(psi, d)
    for i in range(n):
        bops.applySingleSiteOp(psi4, trPs, i)
    renyiSum = bops.getOverlap(psi4, psi4)
    print('renyisum = ' + str(renyiSum))
    return -np.log(renyiSum) / np.log(d) - n


def get2PsiVectorized(psi: List[tn.Node], d: int):
    paulis = basicdefs.getPauliMatrices(d)
    tracePs = np.zeros((d**2, d**2), dtype=complex)
    for i in range(len(paulis)):
        p = paulis[i]
        traceP = p.transpose().reshape([d**2])
        tracePs[i, :] = traceP
    psi2 = []
    for i in range(len(psi)):
        temp = psi[i].tensor.reshape([1] + list(psi[i].shape))
        chiL = psi[i].edges[0].dimension
        chiR = psi[i].edges[2].dimension
        tensorProd = np.tensordot(temp, bops.conj(temp), axes=(0, 0)). \
            transpose([0, 3, 1, 4, 2, 5]).reshape([chiL**2, d**2, chiR**2])
        psi2.append(tn.Node(tensorProd))
    return psi2, tn.Node(tracePs / np.sqrt(d))


def getSecondRenyiFromRandomVecs(psi: List[tn.Node], d: int, outdir='results', rep=1):
    psi2, tracePs = get2PsiVectorized(psi, d)
    for i in range(len(psi)):
        bops.applySingleSiteOp(psi2, tracePs, i)
    M = 1000
    steps = 2**(4 * len(psi))
    try:
        os.mkdir(outdir)
    except FileExistsError:
        pass
    for step in range(steps):
        mySum = 0
        for m in range(M):
            vs = randomVecs.getVs(1, len(psi), d=d ** 2)[0]
            curr = tn.Node(np.eye(1, dtype=complex))
            for site in range(len(psi)):
                curr = bops.multiContraction(curr, psi2[site], '1', '0', cleanOr1=True)
                curr = bops.multiContraction(curr, tn.Node(np.array(vs[site])), '1', '0', cleanOr1=True)
            mySum += curr.tensor[0, 0]**4
        with open(outdir + '/est_' + str(rep) + '_' + str(step), 'wb') as f:
            pickle.dump(mySum / M, f)
        mySum = 0