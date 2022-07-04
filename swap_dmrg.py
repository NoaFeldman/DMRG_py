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
    for i in range(origin, target - np.sign(target - origin), np.sign(target - origin)):
        apply_op(psi, i, swap_op, dir)


def apply_op(psi: List[tn.Node], i: int, op: tn.Node, dir:str):
    M = bops.permute(bops.contract(bops.contract(psi[i], psi[i + 1], '2', '0'), op, '12', '01'), [0, 2, 3, 1])
    [psi[i], psi[i + 1], te] = bops.svdTruncation(M, [0, 1], [2, 3], dir)


def trotter_sweep(psi: List[tn.Node], trotter_terms: List[tn.Node], swap_op: tn.Node):
    for k in range(len(psi) - 1, 0, -1):
        for ni in range(1, len(trotter_terms) + 1):
            swap_site(psi, k - ni, k - 1, swap_op, '>>')
            apply_op(psi, k-1, trotter_terms[ni - 1], '<<')
        for ni in range(1, len(trotter_terms) + 1):
            swap_site(psi, k - ni, k - 1, swap_op, '>>')
    for k in range(len(psi) - 1):
        for ni in range(1, len(trotter_terms) + 1):
            swap_site(psi, k + ni, k + 1, swap_op, '<<')
            apply_op(psi, k, trotter_terms[ni - 1], '>>')
        for ni in range(1, len(trotter_terms) + 1):
            swap_site(psi, k + ni, k + 1, swap_op, '<<')
        psi = bops.shiftWorkingSite(psi, k, '>>')


N = 8
d = 2
psi = [tn.Node(np.array([1, 0, 0, 0]).reshape([1, d**2, 1])) for n in range(N)]
swap_op = tn.Node(np.eye(d**4).reshape([d**2] * 4).transpose([0, 1, 3, 2]))
trotter_sweep(psi, [tn.Node(np.eye(d**4).reshape([d**2] * 4)) for i in range(3)], swap_op)