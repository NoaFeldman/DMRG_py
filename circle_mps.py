import os.path
import basicOperations as bops
import numpy as np
import DMRG as dmrg
import tensornetwork as tn
import pickle
import sys
from os import path


def get_H_terms(N, onsite_term, neighbor_term, d=2):
    onsite_terms = [np.kron(onsite_term, onsite_term) for i in range(int(N / 2) + int(N % 2 == 1))]
    onsite_terms[0] += neighbor_term
    neighbor_terms = [np.kron(neighbor_term.reshape([d] * 4), np.eye(d**2).reshape([d] * 4)).reshape([d**4, d**4])
                      + np.kron(np.eye(d**2).reshape([d] * 4), neighbor_term.reshape([d] * 4)).reshape([d**4, d**4])
                      for i in range(int(N / 2) - 1)]
    if N % 2 == 0:
        onsite_terms[-1] += neighbor_term
    else:
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

X = np.array([[0, 1], [1, 0]])
Z = np.diag([1, -1])
N = int(sys.argv[1])
ising_lambda = float(sys.argv[2])

psi_0 = antiferromagnetic_state(N)
onsite_terms, neighbor_terms = get_H_terms(N, ising_lambda * X, np.kron(Z, Z))

H = np.zeros((2**N, 2**N))
for i in range(N-1):
    H += np.kron(np.eye(2**i), np.kron(np.kron(Z, Z), np.eye(2**(N - 2 - i))))
H += np.kron(Z, np.kron(np.eye(2**(N - 2)), Z))
for i in range(N):
    H += np.kron(np.eye(2**i), np.kron(ising_lambda * X, np.eye(2**(N - 1 - i))))
vals, vecs = np.linalg.eigh(H)
print(min(vals))

gs, E0, trunc_errs = dmrg.DMRG(psi_0, onsite_terms, neighbor_terms, d=4, initial_bond_dim=16)
print(E0)
for i in range(2**N):
    state = [int(c) for c in bin(i).split('b')[1].zfill(N)]
    org_state = []
    for s in range(int(N/2)):
        org_state.append(state[s])
        org_state.append(state[N - s - 1])
    org_state.append(state[int(N/2)])
    org_state.append(0)
    nodes = [np.array(arr).reshape([1, 4, 1]) for arr in [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]]
    mps = [tn.Node(nodes[org_state[2 * s] + 2 * org_state[2 * s + 1]]) for s in range(int(len(org_state) / 2))]
    overlap = np.round(bops.getOverlap(gs, mps), 3)
    if overlap > 0:
        print(state, org_state, overlap, dmrg.stateEnergy(mps, dmrg.getDMRGH(int(N/2) + 1, onsite_terms, neighbor_terms, d=4)))
