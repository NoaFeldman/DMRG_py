import os.path
import basicOperations as bops
import numpy as np
import DMRG as dmrg
import tensornetwork as tn
import pickle
import sys
from os import path


def get_H_terms(N, onsite_term, neighbor_term, d=2):
    onsite_terms = [np.kron(onsite_term, np.eye(d)) + np.kron(np.eye(d), onsite_term)
                    for i in range(int(N / 2))]
    onsite_terms[0] += neighbor_term
    neighbor_terms = [np.kron(neighbor_term.reshape([d] * 4), np.eye(d**2).reshape([d] * 4)).reshape([d**4, d**4])
                      + np.kron(np.eye(d**2).reshape([d] * 4), neighbor_term.reshape([d] * 4)).reshape([d**4, d**4])
                      for i in range(int(N / 2) - 1)]
    if N % 2 == 0:
        onsite_terms[-1] += neighbor_term
    else:
        onsite_terms = onsite_terms + [np.kron(onsite_term, np.eye(d))]
        neighbor_terms = neighbor_terms \
                         + [np.kron(neighbor_term.reshape([d] * 4), np.eye(d**2).reshape([d] * 4)).reshape([d**4, d**4]) + \
                            np.kron(neighbor_term.reshape([d] * 4), np.eye(d**2).reshape([d] * 4)).reshape([d] * 8)
                                .transpose([1, 0, 2, 3, 5, 4, 6, 7]).reshape([d ** 4] * 2)]
    return onsite_terms, neighbor_terms


def antiferromagnetic_state(N):
    if N % 2 == 1:
        basic_mid_site = np.zeros((4, 2, 4), dtype=complex)
        basic_mid_site[0, 0, 1] = 1
        basic_mid_site[1, 1, 0] = 1
        basic_mid_site[0, 0, 2] = 1
        basic_mid_site[2, 0, 3] = 1
        basic_mid_site[1, 1, 3] = 1
        basic_mid_site[3, 1, 2] = 1

        left_site = np.zeros((1, 2**2, 4**2))
        left_site[0, 1 + 0 * 2, 3 + 1 * 4] = 1
        left_site[0, 0 + 1 * 2, 2 + 0 * 4] = 1
        left_site[0, 0 + 0 * 2, 2 + 3 * 4] = 1
        left_site[0, 1 + 1 * 2, 3 + 2 * 4] = 1

        redundant_site = np.zeros((4, 2, 4))
        for di in range(4):
            redundant_site[di, 0, di] = 1
        down = np.copy(basic_mid_site)
        right_site = np.tensordot(down, redundant_site, [2, 0]).transpose([0, 3, 1, 2]).reshape([4**2, 2**2, 1])

        result = [tn.Node(left_site)]
        for i in range(1, int(N/2)):
            down = np.copy(basic_mid_site)
            up = np.copy(basic_mid_site)
            result.append(tn.Node(np.kron(down, up.transpose([2, 1, 0]))))
        result.append(tn.Node(right_site))
    else:
        basic_site = np.zeros((2, 2, 2), dtype=complex)
        basic_site[0, 0, 1] = 1
        basic_site[1, 1, 0] = 1
        result = [tn.Node(bops.contract(tn.Node(np.eye(4)),
                                tn.Node(np.kron(basic_site, basic_site).reshape([4, 4, 2**2, 4**2])),
                                '01', '01').tensor.reshape([1, 2**2, 4**2]))] + \
                 [tn.Node(np.kron(basic_site, basic_site)) for i in range(int(N/2) - 2)] + \
                 [tn.Node(bops.contract(tn.Node(np.kron(basic_site, basic_site).reshape([4**2, 2**2, 4, 4])),
                                tn.Node(np.eye(4)), '23', '01').tensor.reshape([4**2, 2**2, 1]))]
    for i in range(len(result) - 1):
        result[i], result[i+1], te = bops.svdTruncation(bops.contract(result[i], result[i+1], '2', '0'),
                                                        [0, 1], [2, 3], '>>')
    result[-1].tensor /= np.sqrt(bops.getOverlap(result, result))
    return result


def exact_state(N, vec, d=2):
    ten = tn.Node(vec.reshape([1] + [d]*N + [1]))
    mps = [None] * N
    for i in range(N-1):
        [mps[i], ten, te] = bops.svdTruncation(ten, [0, 1], list(range(2, len(ten.shape))), '>>')
    mps[-1] = ten
    if N % 2 == 1:
        zeros_node = tn.Node(np.zeros((mps[int(N/2)].shape[2], d, mps[int(N/2)].shape[2])))
        for i in range(mps[int(N/2)].shape[2]):
            zeros_node.tensor[i, 0, i] = 1
        mps = mps[:int(N/2) + 1] + [zeros_node] + mps[int(N/2) + 1:]
    circle_mps = [tn.Node(np.kron(mps[0].tensor, mps[-1].tensor.transpose([2, 1, 0])))]
    for i in range(1, int(N/2)):
        circle_mps.append(tn.Node(np.kron(mps[i].tensor, mps[-1-i].tensor.transpose([2, 1, 0]))))
    circle_mps.append(tn.Node(bops.contract(mps[int(len(mps)/2) - 1], mps[int(len(mps)/2)], '2', '0')
                .tensor.transpose([0, 3, 1, 2])
                .reshape([mps[int(len(mps)/2) - 1][0].dimension * mps[int(len(mps)/2)][2].dimension, d**2, 1])))
    return circle_mps

X = np.array([[0, 1], [1, 0]])
Z = np.diag([1, -1])

def ground_states_magic(N):
    ising_lambdas = [np.round(0.1 * i, 8) for i in range(1, 31)]

    psi_0 = antiferromagnetic_state(N)
    for ising_lambda in ising_lambdas:
        for i in range(len(psi_0)):
            psi_0[i] = bops.permute(bops.contract(
                psi_0[i], tn.Node(np.kron(np.eye(2) + ising_lambda * X, np.eye(2) + ising_lambda * X)), '1', '0'), [0, 2, 1])
        onsite_terms, neighbor_terms = get_H_terms(N, ising_lambda * X, np.kron(Z, Z))

        gs, E0, trunc_errs = dmrg.DMRG(psi_0, onsite_terms, neighbor_terms, d=4, initial_bond_dim=16, maxBondDim=512,
                                       accuracy=1e-10, silent=False)
        print(E0)
        # split sites so it is consistent with magicRenyi.getRenyiEntropy
        single_site_gs = []
        for i in range(len(gs)):
            [r, l, te] = bops.svdTruncation(tn.Node(gs[i].tensor.reshape([gs[i][0].dimension, 2, 2, gs[i][2].dimension])), [0, 1], [2, 3], '>>')
            single_site_gs.append(r)
            single_site_gs.append(l)
        single_site_gs = single_site_gs[:N]
        relaxed = bops.relaxState(single_site_gs, 4)
        print('<psi|relaxed(psi)> = ' + str(bops.getOverlap(single_site_gs, relaxed)))
        psi_0 = gs

ground_states_magic(11)