import basicOperations as bops
import numpy as np
import DMRG as dmrg
import tensornetwork as tn
import pickle
import basicDefs as basic
import sys
from typing import List
import scipy.linalg as linalg


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


def get_neighbor_trotter_ops(terms: List[np.array], dt, d) -> List[tn.Node]:
    result = [None for i in range(len(terms))]
    for ti in range(len(terms)):
        result[ti] = tn.Node(linalg.expm(-1j * dt * terms[ti] / 2).reshape([d] * 4))
    return result


def get_single_trotter_op(term, dt) -> tn.Node:
    return tn.Node(linalg.expm(-1j * dt / 2 * term))


def trotter_sweep(psi: List[tn.Node], trotter_single_term: tn.Node,
                  trotter_neighbor_terms: List[tn.Node], swap_op: tn.Node):
    for k in range(len(psi) - 1, 0, -1):
        psi[k] = bops.permute(bops.contract(psi[k], trotter_single_term, '1', '0'), [0, 2, 1])
        for ni in range(1, min(k, len(trotter_neighbor_terms)) + 1):
            swap_site(psi, k - ni, k - 1, swap_op, '>>')
            apply_op(psi, k - 1, trotter_neighbor_terms[ni - 1], '>>')
        for ni in range(1, min(k, len(trotter_neighbor_terms)) + 1):
            swap_site(psi, k - ni, k - 1, swap_op, '>>')
        bops.shiftWorkingSite(psi, k, '<<')
    psi[0] = bops.permute(bops.contract(psi[0], trotter_single_term, '1', '0'), [0, 2, 1])
    for k in range(len(psi) - 1):
        psi[k] = bops.permute(bops.contract(psi[k], trotter_single_term, '1', '0'), [0, 2, 1])
        for ni in range(1, min(len(psi) - k - 1, len(trotter_neighbor_terms)) + 1):
            swap_site(psi, k + ni, k + 1, swap_op, '<<')
            apply_op(psi, k, trotter_neighbor_terms[ni - 1], '<<')
        for ni in range(1, min(len(psi) - k - 1, len(trotter_neighbor_terms)) + 1):
            swap_site(psi, k + ni, k + 1, swap_op, '<<')
        bops.shiftWorkingSite(psi, k, '>>')
    psi[len(psi) - 1] = bops.permute(bops.contract(psi[len(psi) - 1], trotter_single_term, '1', '0'), [0, 2, 1])
