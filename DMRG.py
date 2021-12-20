import tensornetwork as tn
import numpy as np
import basicOperations as bops
import math
from typing import Any, Dict, List, Optional, Set, Text, Tuple, Union, \
    Sequence, Iterable, Type
import pickle


class HOp:
    def __init__(self, singles, r2l, l2r):
        self.singles = singles
        self.r2l = r2l
        self.l2r = l2r

# onsightTerm as a d*d matrix 0, 1
# neighborTerm as a d^2*d^2 matrix 00, 01, 10, 11
def getDMRGH(N, onsiteTerms, neighborTerms, d=2):
    hSingles = [None] * N
    for i in range(N):
        hSingles[i] = tn.Node(onsiteTerms[i], name=('single' + str(i)), axis_names=['s' + str(i) + '*', 's' + str(i)])
    hr2l = [None] * (N)
    hl2r = [None] * (N)
    for i in range(N-1):
        if d == 2:
            neighborTerm = np.reshape(neighborTerms[i], (2, 2, 2, 2))
        elif d == 3:
            neighborTerm = np.reshape(neighborTerms[i], (3, 3, 3, 3))
        else:
            print(" d unsupported!")
            return -1
        pairOp = tn.Node(neighborTerm,
                         axis_names=['s' + str(i) + '*', 's' + str(i+1) + '*', 's' + str(i), 's' + str(i+1)])
        splitted = tn.split_node(pairOp, [pairOp[0], pairOp[2]], [pairOp[1], pairOp[3]],
                                          left_name=('l2r' + str(i)), right_name=('r2l' + str(i) + '*'), edge_name='m')
        splitted[1].reorder_axes([1, 2, 0])
        hr2l[i + 1] = splitted[1]
        hl2r[i] = splitted[0]
    return HOp(hSingles, hr2l, hl2r)

class HExpValMid:
    def __init__(self, opSum, openOp):
    # HLR.opSum is
    # H(1).single x I x I... + I x H(2).single x I... + H.l2r(1) x H.r2l(2) x I...(two degree tensor)
    # HLR.openOp is
    # I x I x...x H(l).l2r(three degree tensor)
        self.opSum = opSum
        self.openOp = openOp


# Returns <H> for the lth site:
#  If l is in the begining of the chain, returns
#   _
#  | |--
#  | |
#  | |-- for dir='>>', and the miror QSpace for dir = '<<'
#  | |
#  |_|--
#
#  else, performs
#   _        _
#  | |--  --| |--
#  | |      | |
#  | |--  --| |--
#  | |      | |
#  |_|--  --|_|--
def getHLR(psi, l, H, dir, HLRold):
    if dir == '>>':
        if l == -1:
            opSum = tn.Node(np.zeros((psi[0].get_dimension(0), psi[0].get_dimension(0)), dtype=complex))
            openOp = tn.Node(np.zeros((psi[0].get_dimension(0), psi[0].get_dimension(0)), dtype=complex))
            return HExpValMid(opSum, openOp)
        else:
            psil = bops.copyState([psi[l]], conj=False)[0]
            psilConj = bops.copyState([psi[l]], conj=True)[0]
            singlel = bops.copyState([H.singles[l]], conj=False)[0]
            psil[0] ^ psilConj[0]
            psil[1] ^ singlel[0]
            psilConj[1] ^ singlel[1]
            opSum1 = tn.contract_between(psil, \
                     tn.contract_between(singlel, psilConj), name='operator-sum')
            if l > 0:
                psil = bops.copyState([psi[l]], conj=False)[0]
                psilConj = bops.copyState([psi[l]], conj=True)[0]
                HLRoldCopy = bops.copyState([HLRold.openOp])[0]
                r2l_l = bops.copyState([H.r2l[l]], conj=False)[0]
                psil[0] ^ HLRoldCopy[0]
                psilConj[0] ^ HLRoldCopy[1]
                psil[1] ^ r2l_l[0]
                psilConj[1] ^ r2l_l[1]
                r2l_l[2] ^ HLRoldCopy[2]
                opSum2 = tn.contract_between(psil, tn.contract_between(psilConj, tn.contract_between(r2l_l, HLRoldCopy)))
                opSum1 = bops.addNodes(opSum1, opSum2)

            psil = bops.copyState([psi[l]], conj=False)[0]
            psilConj = bops.copyState([psi[l]], conj=True)[0]
            HLRoldCopy = bops.copyState([HLRold.opSum])[0]
            psil[0] ^ HLRoldCopy[0]
            psilConj[0] ^ HLRoldCopy[1]
            psil[1] ^ psilConj[1]
            opSum3 = tn.contract_between(psil, tn.contract_between(psilConj, HLRoldCopy))
            opSum1 = bops.addNodes(opSum1, opSum3)

            if l < len(psi) - 1:
                psil = bops.copyState([psi[l]], conj=False)[0]
                psilConj = bops.copyState([psi[l]], conj=True)[0]
                l2r_l = bops.copyState([H.l2r[l]], conj=False)[0]
                psil[0] ^ psilConj[0]
                psil[1] ^ l2r_l[0]
                psilConj[1] ^ l2r_l[1]
                openOp = tn.contract_between(psil, tn.contract_between(psilConj, l2r_l), name='open-operator')
            else:
                openOp = None
            return HExpValMid(opSum1, openOp)
    if dir == '<<':
        if l == len(psi):
            opSum = tn.Node(np.zeros((psi[l-1].get_dimension(2), psi[l-1].get_dimension(2)), dtype=complex))
            openOp = tn.Node(np.zeros((psi[l-1].get_dimension(2), psi[l-1].get_dimension(2)), dtype=complex))
            return HExpValMid(opSum, openOp)
        else:
            psil = bops.copyState([psi[l]], conj=False)[0]
            psilConj = bops.copyState([psi[l]], conj=True)[0]
            single_l = bops.copyState([H.singles[l]], conj=False)[0]
            psil[2] ^ psilConj[2]
            psil[1] ^ single_l[0]
            psilConj[1] ^ single_l[1]
            opSum1 = tn.contract_between(psil, \
                                         tn.contract_between(single_l, psilConj), name='operator-sum')

            if l < len(psi) -1:
                psil = bops.copyState([psi[l]], conj=False)[0]
                psilConj = bops.copyState([psi[l]], conj=True)[0]
                HLRoldCopy = bops.copyState([HLRold.openOp])[0]
                l2r_l = bops.copyState([H.l2r[l]], conj=False)[0]
                psil[2] ^ HLRoldCopy[0]
                psilConj[2] ^ HLRoldCopy[1]
                psil[1] ^ l2r_l[0]
                psilConj[1] ^ l2r_l[1]
                l2r_l[2] ^ HLRoldCopy[2]
                opSum2 = tn.contract_between(psil, tn.contract_between(psilConj, tn.contract_between(l2r_l, HLRoldCopy)))
                opSum1 = bops.addNodes(opSum1, opSum2)

            psil = bops.copyState([psi[l]], conj=False)[0]
            psilConj = bops.copyState([psi[l]], conj=True)[0]
            HLRoldCopy = bops.copyState([HLRold.opSum])[0]
            psil[2] ^ HLRoldCopy[0]
            psilConj[2] ^ HLRoldCopy[1]
            psil[1] ^ psilConj[1]
            opSum3 = tn.contract_between(psil, tn.contract_between(psilConj, HLRoldCopy))
            opSum1 = bops.addNodes(opSum1, opSum3)

            if l > 0:
                psil = bops.copyState([psi[l]], conj=False)[0]
                psilConj = bops.copyState([psi[l]], conj=True)[0]
                r2l_l = bops.copyState([H.r2l[l]], conj=False)[0]
                psil[2] ^ psilConj[2]
                psil[1] ^ r2l_l[0]
                psilConj[1] ^ r2l_l[1]
                openOp = tn.contract_between(psil, tn.contract_between(psilConj, r2l_l), name='open-operator')
            else:
                openOp = None
            return HExpValMid(opSum1, openOp)


# k is the working site
def lanczos(HL, HR, H, k, psi, psiCompare):
    [T, base] = getTridiagonal(HL, HR, H, k, psi, psiCompare)
    [Es, Vs] = np.linalg.eigh(T)
    minIndex = np.argmin(Es)
    E0 = Es[minIndex]
    M = None
    for i in range(len(Es)):
        M = bops.addNodes(M, bops.multNode(base[i], Vs[i][minIndex]))

    M = bops.multNode(M, 1/bops.getNodeNorm(M))
    return [M, E0]

def getIdentity(psi, k, dir):
    psil = bops.copyState([psi[k]])[0]
    psilCopy = bops.copyState([psi[k]], conj=True)[0]
    if dir == '>>':
        result = bops.multiContraction(psil, psilCopy, '01', '01').tensor
    else:
        result = bops.multiContraction(psil, psilCopy, '12', '12').tensor
    for i in range(len(result)):
        for j in range(len(result[0])):
            result[i][j] = round(result[i][j], 2)
    return result


def getTridiagonal(HL, HR, H, k, psi, psiCompare=None):
    accuracy = 1e-17 # 1e-12

    v = bops.multiContraction(psi[k], psi[k + 1], '2', '0')
    # Small innaccuracies ruin everything!
    v.set_tensor(v.get_tensor() / bops.getNodeNorm(v))

    psiCopy = bops.copyState(psi)

    basis = []
    basis.append(bops.copyState([v])[0])
    Hv = applyHToM(HL, HR, H, v, k)
    alpha = bops.multiContraction(v, Hv, '0123', '0123*').get_tensor()

    if psiCompare is not None:
        copyV = bops.copyState([v])[0]
        psiCopy = bops.assignNewSiteTensors(psiCopy, k, copyV, '>>')[0]
    E = stateEnergy(psi, H)
    w = bops.addNodes(Hv, bops.multNode(v, -alpha))
    beta = bops.getNodeNorm(w)
    # Start with T as an array and turn into tridiagonal matrix at the end.
    Tarr = [[0, 0, 0]]
    Tarr[0][1] = alpha
    counter = 0
    formBeta = 2 * beta # This is just some value to init formBeta > beta.
    while (beta > accuracy) and (counter <= 50) and (beta < formBeta):
        Tarr[counter][2] = beta
        Tarr.append([0, 0, 0])
        Tarr[counter + 1][0] = beta
        counter += 1

        v = bops.multNode(w, 1 / beta)
        basis.append(bops.copyState([v])[0])

        if psiCompare is not None:
            copyV = bops.copyState([v])[0]
            psiCopy = bops.assignNewSiteTensors(psiCopy, k, copyV, '>>')[0]
        Hv = applyHToM(HL, HR, H, v, k)

        alpha = bops.multiContraction(v, Hv, '0123', '0123*').get_tensor()
        Tarr[counter][1] = alpha
        w = bops.addNodes(bops.addNodes(Hv, bops.multNode(v, -alpha)), \
                          bops.multNode(bops.copyState([basis[counter - 1]])[0], -beta), cleanOr2=True)
        formBeta = beta
        beta = bops.getNodeNorm(w)
    T = np.zeros((len(Tarr), len(Tarr)), dtype=complex)
    T[0][0] = Tarr[0][1]
    if len(Tarr) > 1:
        T[0][1] = Tarr[0][2]
    for i in range(1, len(Tarr)-1):
        T[i][i-1] = Tarr[i][0]
        T[i][i] = Tarr[i][1]
        T[i][i+1] = Tarr[i][2]
    T[len(Tarr)-1][len(Tarr)-2] = Tarr[len(Tarr)-1][0]
    T[len(Tarr) - 1][len(Tarr) - 1] = Tarr[len(Tarr) - 1][1]
    return [T, basis]


def applyHToM(HL, HR, H, M, k):
    k1 = k
    k2 = k + 1

    # Add HL.opSum x h.identity(k1) x h.identity(k2) x I(Right)
    # and I(Left) x h.identity(k1) x h.identity(k2) x HR.opSum
    Hv = bops.multiContraction(HL.opSum, M, '0', '0')
    Hv = bops.addNodes(Hv, bops.multiContraction(M, HR.opSum, '3', '0'))

    # Add I(Left) x h.single(k1) x h.identity(k2) x I(Right)
    # And I(Left) x h.identity(k1) x h.single(k2) x I(Right)
    Hv = bops.addNodes(Hv, bops.multiContraction(M, H.singles[k1], '1', '0').reorder_axes([0, 3, 1, 2]))
    Hv = bops.addNodes(Hv, bops.multiContraction(M, H.singles[k2], '2', '0').reorder_axes([0, 1, 3, 2]))

    # Add HL.openOp x h.r2l(k1) x h.identity(k2) x I(Right)
    # And I(Left) x h.identity(k1) x h.l2r(k2) x HR.openOp
    HK1R2L = bops.multiContraction(M, H.r2l[k1], '1', '0')
    if HK1R2L is not None:
        HK1R2L.reorder_axes([0, 4, 3, 1, 2])
    Hv = bops.addNodes(Hv, bops.multiContraction(HL.openOp, HK1R2L, '02', '01'))
    HK2L2R = bops.multiContraction(M, H.l2r[k2], '2', '0')
    if HK2L2R is not None:
        HK2L2R.reorder_axes([0, 1, 3, 4, 2])
    Hv = bops.addNodes(Hv, bops.multiContraction(HK2L2R, HR.openOp, '43', '02'))

    # Add I(Left) x h.l2r(k1) x h.r2l(k2) x I(Right)
    HK1K2 = bops.multiContraction(M, H.l2r[k1], '1', '0')
    HK1K2 = bops.multiContraction(HK1K2, H.r2l[k2], '14', '02').reorder_axes([0, 2, 3, 1])
    Hv = bops.addNodes(Hv, HK1K2)

    return Hv


def dmrgStep(HL, HR, H, psi, k, dir, psiCompare=None, opts=None, maxBondDim=128):
    # Perform a single DMRG step:
    # 1. Contracts psi(k) and psi(k + dir) to get M.
    # 2. Performs lancsoz and get a new contracted M.
    # 3. Performs an SVD in order to get the new working site, at k + dir.
    # 4. Calculates HL(k) / HR(k) (according to dir)
    k1 = k
    k2 = k + 1
    [M, E0] = lanczos(HL, HR, H, k1, psi, psiCompare)
    [psi, truncErr] = bops.assignNewSiteTensors(psi, k, M, dir, maxBondDim=maxBondDim)
    if dir == '>>':
        if psiCompare is not None:
            for state in psiCompare:
                state = bops.shiftWorkingSite(state, k, '>>')
                psi = bops.getOrthogonalState(state, psiInitial=psi)
            HLs, HRs = getHLRs(H, psi, workingSite=k+1)
            return psi, HLs, HRs, E0, truncErr
        else:
            newHL = getHLR(psi, k, H, dir, HL)
            return psi, newHL, E0, truncErr
    else:
        if psiCompare is not None:
            for state in psiCompare:
                state = bops.shiftWorkingSite(state, k, '<<')
                psi = bops.getOrthogonalState(state, psiInitial=psi)
            HLs, HRs = getHLRs(H, psi, workingSite=k)
            return psi, HLs, HRs, E0, truncErr
        else:
            newHR = getHLR(psi, k+1, H, dir, HR)
            return psi, newHR, E0, truncErr


# Assume the OC is at the last (rightmost) site. sweeps all the way left and back right again.
def dmrgSweep(psi, H, HLs, HRs, psiCompare=None, maxBondDim=128):
    k = len(psi) - 2
    maxTruncErr = 0
    while k > 0:
        if psiCompare is None:
            if k == 13:
                b = 1
            [psi, newHR, E0, truncErr] = dmrgStep(HLs[k], HRs[k+2], H, psi, k, '<<', psiCompare, maxBondDim=maxBondDim)
            # if HRs[k+1] is not None:
            # TODO remove all nodes in HLR
                # tn.remove_node(HRs[k+1])
            HRs[k+1] = newHR
        else:
            [psi, HLs, HRs, E0, truncErr] = dmrgStep(HLs[k], HRs[k+2], H, psi, k, '<<', psiCompare, maxBondDim=maxBondDim)
        if len(truncErr) > 0 and maxTruncErr < max(truncErr):
            maxTruncErr = max(truncErr)
        k -= 1
    for k in range(len(psi) - 2):
        E0Old = E0
        if psiCompare is None:
            [psi, newHL, E0, truncErr] = dmrgStep(HLs[k], HRs[k + 2], H, psi, k, '>>', psiCompare, maxBondDim=maxBondDim)
            HLs[k + 1] = newHL
        else:
            [psi, HLs, HRs, E0, truncErr] = dmrgStep(HLs[k], HRs[k + 2], H, psi, k, '>>', psiCompare, maxBondDim=maxBondDim)
        # if E0 > E0Old:
        #     print('E0 > E0Old, k = ' + str(k) + ', E0Old = ' + str(E0Old) + ', E0 = ' + str(E0))
        # if HLs[k+1] is not None:
        # TODO remove all nodes in HLR
        #     tn.remove_node(HLs[k+1])
        if len(truncErr) > 0 and maxTruncErr < max(truncErr):
            maxTruncErr = max(truncErr)
    return psi, E0, truncErr, HLs, HRs


def getHLRs(H, psi, workingSite=None):
    N = len(psi)
    if workingSite is None:
        workingSite=N-1
    HLs = [None] * (N + 1)
    HLs[0] = getHLR(psi, -1, H, '>>', 0)
    for l in range(workingSite):
        HLs[l + 1] = getHLR(psi, l, H, '>>', HLs[l])
    HRs = [None] * (N + 1)
    HRs[N] = getHLR(psi, N, H, '<<', 0)
    l = N-1
    while l > workingSite:
        HRs[l] = getHLR(psi, l, H, '<<', HRs[l+1])
        l -= 1
    return HLs, HRs


def getGroundState(H, HLs, HRs, psi, psiCompare=None, accuracy=10**(-12), maxBondDim=256, initial_bond_dim=2):
    truncErrs = []
    bondDim = initial_bond_dim
    [psi, E0, truncErr, HLs, HRs] = dmrgSweep(psi, H, HLs, HRs, psiCompare, maxBondDim=maxBondDim)
    truncErrs.append(truncErr)
    for i in range(500):
        print(E0)
        [psi, ECurr, truncErr, HLs, HRs] = dmrgSweep(psi, H, HLs, HRs, psiCompare, maxBondDim=bondDim)
        truncErrs.append(truncErr)
        if np.abs((ECurr - E0) / E0) < accuracy:
            return psi, ECurr, truncErrs
        if (i+1) % 10 == 0:
            bondDim = min(bondDim * 2, maxBondDim)
            if bondDim == 4096:
                print(4096)
        E0 = ECurr
    print('DMRG: Sweeped for 500 times and still did not converge.')


def stateEnergy(psi: List[tn.Node], H: HOp):
    E = 0
    for i in range(len(psi)):
        psiCopy = bops.copyState(psi)
        single_i = bops.copyState([H.singles[i]])[0]
        psiCopy[i] = tn.contract(psiCopy[i][1] ^ single_i[0], name=('site' + str(i))).reorder_axes([0, 2, 1])
        E += bops.getOverlap(psiCopy, psi)
        bops.removeState(psiCopy)
        tn.remove_node(single_i)
    for i in range(len(psi) - 1):
        psiCopy = bops.copyState(psi)
        r2l = bops.copyState([H.r2l[i+1]])[0]
        l2r = bops.copyState([H.l2r[i]])[0]
        psiCopy[i][2] ^ psiCopy[i+1][0]
        psiCopy[i][1] ^ l2r[0]
        r2l[0] ^ psiCopy[i+1][1]
        l2r[2] ^ r2l[2]
        M = tn.contract_between(psiCopy[i], \
                                tn.contract_between(l2r, tn.contract_between(r2l, psiCopy[i+1])))
        if bops.multiContraction(M, M, '0123', '0123*').tensor != 0:
            [psiCopy, te] = bops.assignNewSiteTensors(psiCopy, i, M, '>>')
            E += bops.getOverlap(psiCopy, psi)
        bops.removeState(psiCopy)
        tn.remove_node(r2l)
        tn.remove_node(l2r)
    return E


def getidx(N, q):
    res = []
    for i in range(2**N):
        if q == N - 2 * bin(i).count("1"):
            res.append(i)
    return np.array(res).reshape(len(res), 1)


def DMRG(psi0, onsiteTerms, neighborTerms, d=2, maxBondDim=256, accuracy=1e-12, initial_bond_dim=2):
    H = getDMRGH(len(psi0), onsiteTerms, neighborTerms, d=d)
    psi0Copy = bops.copyState(psi0)
    HLs, HRs = getHLRs(H, psi0Copy)
    return getGroundState(H, HLs, HRs, psi0Copy, maxBondDim=maxBondDim, accuracy=accuracy, initial_bond_dim=initial_bond_dim)


example = False
if example:
    XX = np.zeros((4, 4), dtype=complex)
    XX[1, 2] = 1
    XX[2, 1] = 1
    Z = np.zeros((2, 2), dtype=complex)
    Z[0, 0] = 1
    Z[1, 1] = -1
    N = 8
    exactH = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(N-1):
        curr = np.eye(1, dtype=complex)
        for j in range(i):
            curr = np.kron(curr, np.eye(2))
        curr = np.kron(curr, XX)
        for j in range(i+1, N-1):
            curr = np.kron(curr, np.eye(2))
        exactH += curr
    idx1 = getidx(N, - N + 2)
    exactH1 = exactH[idx1, idx1.T]
    H = getDMRGH(N, [np.copy(Z) * 0 for i in range(N)], [np.copy(XX) for i in range(N-1)])
    psi0 = bops.getStartupState(N, mode='antiferromagnetic')
    HLs, HRs = getHLRs(H, psi0)
    gs, E0, truncErrs = getGroundState(H, HLs, HRs, psi0)
    with open('results/psiXX_' + str(N), 'wb') as f:
        pickle.dump(gs, f)

