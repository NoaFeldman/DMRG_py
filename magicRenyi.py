import tensornetwork as tn
import numpy as np
from typing import List
import basicOperations as bops
import pickle
import os
import randomUs as ru
from basicOperations import dagger
import basicDefs

# Vectorize all matrices. trPs, trPDaggers are built such that applying them to the vectorized DM we get the vector
# (tr(\rho), tr(X\rho), tr(Y\rho), tr(Z\rho)), or its eqiovalent in general d.
# the state returned is a state with each site representing a tensor product of 4 copies of the state,
# two regular and two conjugated.
def get4PsiVectorized(psi: List[tn.Node], d: int):
    paulis = basicDefs.getPauliMatrices(d)
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
    return -np.log(renyiSum) / np.log(d) - n


def getSecondRenyi_optimizedBasis(psi, d, opt='min'):
    if opt == 'min':
        best_result = 100
    else:
        best_result = 0
    best_basis = None
    thetas = [0.1 * np.pi * i for i in range(6)]
    phis = [0.1 * np.pi * i for i in range(6)]
    etas = [0.1 * np.pi * i for i in range(6)]
    for ti in range(len(thetas)):
        theta = thetas[ti]
        u_theta = ru.getUTheta(theta, d)
        for pi in range(len(phis)):
            phi = phis[pi]
            u_phi = ru.getUPhi(phi, d)
            for ei in range(len(etas)):
                eta = etas[ei]
                u_eta = ru.getUEta(eta, d)
                u = tn.Node(np.matmul(u_eta, np.matmul(u_theta, u_phi)))
                psi_copy = bops.copyState(psi)
                for site in range(len(psi)):
                    psi_copy[site] = bops.permute(bops.contract(psi_copy[site], u, '1', '0'), [0, 2, 1])
                curr = getSecondRenyi(psi_copy, d)
                if opt == 'min':
                    if curr < best_result:
                        best_result = curr
                        best_basis = [theta, phi, eta]
                else:
                    if curr > best_result:
                        best_result = curr
                        best_basis = [theta, phi, eta]
    return best_result, best_basis


def tensorKetBra(tensorKet, chiL, chiR, d, tensorBra=None):
    if tensorBra is None:
        tensorBra = tensorKet
    return np.outer(tensorKet, np.conj(tensorBra)).reshape([chiL, d, chiR, chiL, d, chiR]) \
        .transpose([0, 3, 1, 4, 2, 5]).reshape([chiL ** 2, d ** 2, chiR ** 2])


def ketBra(site: tn.Node, d):
    chiL = int(site[0].dimension)
    chiR = int(site[2].dimension)
    tensorProd = tensorKetBra(site.tensor, chiL, chiR, d)
    # tensorProd = np.tensordot(temp, np.conj(temp), axes=(0, 0)). \
    #     transpose([0, 3, 1, 4, 2, 5]).reshape([chiL ** 2, d ** 2, chiR ** 2])
    return tensorProd


def get2PsiVectorized(psi: List[tn.Node], d: int):
    paulis = basicDefs.getPauliMatrices(d)
    tracePs = np.zeros((d**2, d**2), dtype=complex)
    for i in range(len(paulis)):
        p = paulis[i]
        traceP = p.transpose().reshape([d**2])
        tracePs[i, :] = traceP
    psi2 = [None for i in range(len(psi))]
    for i in range(len(psi)):
        psi2[i] = tn.Node(ketBra(psi[i], d))
    return psi2, tn.Node(tracePs / np.sqrt(d))


def getSecondRenyiFromRandomVecs(psi: List[tn.Node], d: int, outdir='results', rep=1, speedup=False, rank=2):
    psi2, tracePs = get2PsiVectorized(psi, d)
    for i in range(len(psi)):
        bops.applySingleSiteOp(psi2, tracePs, i)
    M = 1000
    steps = 1000 # d**(2 * len(psi))
    try:
        os.mkdir(outdir)
    except FileExistsError:
        pass
    for step in range(steps):
        mySum = 0
        for m in range(M):
            if not speedup:
                vs = [np.exp(1j * np.pi * np.random.randint(4, size=d**2) / 2) for site in range(len(psi))] #randomVecs.getVs(1, len(psi), d=d**2)[0]
            else:
                vs = [np.exp(1j * np.pi * np.random.randint(4, size=d ** 4) / 2) for site in range(int(len(psi)/2))]
            curr = tn.Node(np.eye(1, dtype=complex))
            currDagger = tn.Node(np.eye(1, dtype=complex))
            if not speedup:
                for siteInd in range(len(psi)):
                    # if speedup:
                        # chiL = int(psi2[siteInd][0].dimension)
                        # chiR = int(psi2[siteInd][2].dimension)
                        # siteTensor = np.zeros((chiL**2, d**2, chiR**2), dtype=complex)
                        # for i in range(d**2):
                        #     siteTensor[:, i, :] = tensorKetBra(psi2[siteInd].tensor[:, i, :].reshape([chiL, 1, chiR]),
                        #                                        chiL, chiR, 1).reshape([chiL**2, chiR**2])
                        # site = tn.Node(siteTensor)
                    # else:
                    vTensor = np.array(vs[siteInd])
                    v = tn.Node(vTensor)
                    site = psi2[siteInd]
                    currDagger = bops.multiContraction(currDagger, site, '1', '0*', cleanOr1=True)
                    currDagger = bops.multiContraction(currDagger, v, '1', '0', cleanOr1=True)
                    curr = bops.multiContraction(curr, site, '1', '0', cleanOr1=True, cleanOr2=True)
                    curr = bops.multiContraction(curr, v, '1', '0', cleanOr1=True, cleanOr2=True)
                # if speedup:
                #     mySum += np.abs(curr.tensor[0, 0] ** 2)
                # else:
            else:
                for siteInd in range(int(len(psi2)/2)):
                    vTensor = np.array(vs[siteInd])
                    v = tn.Node(vTensor)
                    site = bops.unifyLegs(bops.multiContraction(psi2[2 * siteInd], psi2[2 * siteInd + 1], '2', '0'),
                                          1, 2, cleanOriginal=True)
                    currDagger = bops.multiContraction(currDagger, site, '1', '0*', cleanOr1=True)
                    currDagger = bops.multiContraction(currDagger, v, '1', '0', cleanOr1=True)
                    curr = bops.multiContraction(curr, site, '1', '0', cleanOr1=True, cleanOr2=True)
                    curr = bops.multiContraction(curr, v, '1', '0', cleanOr1=True, cleanOr2=True)
            mySum += curr.tensor[0, 0] ** rank * currDagger.tensor[0, 0] ** rank
        with open(outdir + '/est_' + str(rep) + '_' + str(step), 'wb') as f:
            pickle.dump(mySum / M, f)
            # print(mySum / M)
        mySum = 0


digs = '0123456789'
def int2base(x, base, N=None):
    if x == 0:
        res = '0'
    digits = []
    while x:
        digits.append(digs[int(x % base)])
        x = int(x / base)
    digits.reverse()
    res = ''.join(digits)
    if N is None:
        return res
    return '0' * (N - len(res)) + res


def getPOp(index, paulis, d, n):
    pauliString = int2base(index, np.square(d), N=n)
    op = np.eye(1)
    for char in pauliString:
        op = np.kron(op, paulis[int(char)])
    return op


def getSecondRenyiExact(psi: List[tn.Node], d: int):
    n = len(psi)
    dm = tn.Node(np.eye(1))
    for i in range(n - 1):
        dm = bops.multiContraction(bops.multiContraction(psi[i], dm, '0', '1'), psi[i], [2 * i + 2], '0*')
    dm = bops.multiContraction(bops.multiContraction(psi[n - 1], dm, '0', '1'), psi[n - 1], [2 * n, 1], '02*')
    dm = dm.tensor.transpose(list(range(n)) + list(range(2*n - 1, n-1, -1))).reshape([d**n, d**n])
    return getSecondRenyiExact_dm(dm, d)


def getSecondRenyiExact_dm(dm, d):
    n = int(np.log(len(dm)) / np.log(d))
    paulis = basicDefs.getPauliMatrices(d)
    renyiSum = 0
    for i in range(d**(2 * n)):
        pOp = getPOp(i, paulis, d, n)
        renyiSum += np.abs(np.matmul(dm, pOp).trace())**4 / d**(2*n)
    print('renyi sum = ' + str(renyiSum))
    return -np.log(renyiSum) / np.log(d) - n


def getHalfRenyiExact_dm(dm, d, paulis=None):
    n = int(np.log(len(dm)) / np.log(d))
    if paulis is None:
        paulis = basicDefs.getPauliMatrices(d)
    renyiSum = 0
    for i in range(d**(2 * n)):
        pOp = getPOp(i, paulis, d, n)
        if np.abs(np.matmul(dm, pOp).trace()) / d**n > 1e-10:
            b = 1
        renyiSum += np.abs(np.matmul(dm, pOp).trace()) / d**n
    return np.log(renyiSum)


def getHalfRenyiExact_dm_optimized(dm, d):
    best_result = 100
    worst_result = 0
    best_basis = None
    worst_basis = None
    thetas = [0.25 * np.pi * i for i in range(3)]
    phis = [0.25 * np.pi * i for i in range(3)]
    etas = [0.25 * np.pi * i for i in range(3)]
    for ti in range(len(thetas)):
        theta = thetas[ti]
        u_theta = ru.getUTheta(theta, d)
        for pi in range(len(phis)):
            phi = phis[pi]
            u_phi = ru.getUPhi(phi, d)
            for ei in range(len(etas)):
                eta = etas[ei]
                u_eta = ru.getUEta(eta, d)
                u = np.matmul(u_eta, np.matmul(u_theta, u_phi))
                paulis = [np.matmul(u, np.matmul(p, dagger(u))) for p in basicDefs.getPauliMatrices(d)]
                curr = getHalfRenyiExact_dm(dm, d, paulis)
                if curr < best_result:
                    best_result = curr
                    best_basis = [theta, phi, eta]
                if curr > worst_result:
                    worst_result = curr
                    worst_basis = [theta, phi, eta]
    return best_result, best_basis, worst_result, worst_basis
