import basicOperations as bops
import numpy as np
import DMRG as dmrg
import tensornetwork as tn
import pickle
import basicDefs as basic
import sys
from typing import List
import scipy.linalg as linalg
import superradiance


def swap_site(psi: List[tn.Node], origin: int, target: int, swap_op: tn.Node, dir: str):
    if target == origin:
        return
    for i in range(origin, target, np.sign(target - origin)):
        if dir == '>>':
            apply_op(psi, i, swap_op, dir)
        else:
            apply_op(psi, i - 1, swap_op, dir)


def apply_op(psi: List[tn.Node], i: int, op: tn.Node, dir:str):
    M = bops.permute(bops.contract(bops.contract(psi[i], psi[i + 1], '2', '0'), op, '12', '01'), [0, 2, 3, 1])
    [psi[i], psi[i + 1], te] = bops.svdTruncation(M, [0, 1], [2, 3], dir)


def trotter_sweep(psi: List[tn.Node], trotter_single_term: tn.Node,
                  trotter_neighbor_terms: List[tn.Node], swap_op: tn.Node):
    for k in range(len(psi) - 1, 0, -1):
        psi[k] = bops.permute(bops.contract(psi[k], trotter_single_term, '1', '0'), [0, 2, 1])
        for ni in range(1, min(k, len(trotter_neighbor_terms)) + 1):
            swap_site(psi, k - ni, k - 1, swap_op, '>>')
            apply_op(psi, k - 1, trotter_neighbor_terms[ni - 1], '<<')
        for ni in range(1, min(k, len(trotter_neighbor_terms)) + 1):
            swap_site(psi, k - ni, k - 1, swap_op, '>>')
    for k in range(len(psi) - 1):
        psi[k] = bops.permute(bops.contract(psi[k], trotter_single_term, '1', '0'), [0, 2, 1])
        for ni in range(1, min(len(psi) - k - 1, len(trotter_neighbor_terms)) + 1):
            swap_site(psi, k + ni, k + 1, swap_op, '<<')
            apply_op(psi, k, trotter_neighbor_terms[ni - 1], '>>')
        for ni in range(1, min(len(psi) - k - 1, len(trotter_neighbor_terms)) + 1):
            swap_site(psi, k + ni, k + 1, swap_op, '<<')


def get_neighbor_trotter_ops(terms: List[np.array], dt) -> List[tn.Node]:
    result = [None for i in range(len(terms))]
    for ti in range(len(terms)):
        result[ti] = tn.Node(linalg.expm(1j * dt * terms[ti] / 2).reshape([d**2] * 4))
    return result


N = 4
d = 2
k = 2 * np.pi / 10
theta = 0
nn_num = 3
Gamma = 1
Omega = 0.0 #float(sys.argv[3]) / Gamma
sigma = superradiance.sigma
case = 'dicke'
T = 10
timesteps = T * 100
dt = T / timesteps

Deltas, gammas = superradiance.get_gnm(Gamma, k, theta, nn_num, case)
single = superradiance.get_single_L_term(Omega, Gamma, sigma)
trotter_single_op = tn.Node(linalg.expm(dt * single / 2))
pairs = superradiance.get_pair_L_terms(Deltas, gammas, nn_num, sigma)
terms = []
for ni in range(nn_num):
    curr = np.kron(pairs[0][0][ni], pairs[0][1])
    for pi in range(1, len(pairs)):
        curr += np.kron(pairs[pi][0][ni], pairs[pi][1])
    terms.append(curr)
neighbor_trotter_ops = get_neighbor_trotter_ops(terms, -1j * dt)

psi = [tn.Node(np.array([1, 0, 0, 0]).reshape([1, d**2, 1])) for n in range(N)]
swap_op = tn.Node(np.eye(d**4).reshape([d**2] * 4).transpose([0, 1, 3, 2]))
J_expect = np.zeros(timesteps)
for ti in range(timesteps):
    print('--')
    print(ti)
    for si in range(N):
        J_expect[ti] += bops.getOverlap(psi,
                                        [tn.Node(superradiance.I) for i in range(si)] + [
                                            tn.Node(np.matmul(sigma.T, sigma).reshape([1, d ** 2, 1]))]
                                        + [tn.Node(superradiance.I) for i in range(si + 1, N)])
        for sj in range(N):
            if si != sj:
                J_expect[ti] += bops.getOverlap(psi,
                                                [tn.Node(superradiance.I) for i in range(si)] + [tn.Node(sigma.T.reshape([1, d ** 2, 1]))]
                                                + [tn.Node(superradiance.I) for i in range(si + 1, sj)] + [
                                                    tn.Node(sigma.reshape([1, d ** 2, 1]))]
                                                + [tn.Node(superradiance.I) for i in range(sj + 1, N)])
    trotter_sweep(psi, trotter_single_op, neighbor_trotter_ops, swap_op)
import matplotlib.pyplot as plt
plt.plot(J_expect)
plt.show()