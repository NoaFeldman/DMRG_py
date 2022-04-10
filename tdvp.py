import basicOperations as bops
import numpy as np
import DMRG as dmrg
import tensornetwork as tn
import pickle
import basicDefs as basic
import matplotlib.pyplot as plt
import magicRenyi
import sys
from typing import List
import scipy.linalg as linalg

# k is OC, pair is [k, k+1]
def tdvp_step(psi: List[tn.Node], H: List[tn.Node], k: int,
              projectors_left: List[tn.Node], projectors_right: List[tn.Node], dir: str, dt):
    M = bops.contract(psi[k], psi[k+1], '2', '0')
    # TODO add Krylov here at some point
    H_effective = bops.permute(bops.contract(bops.contract(bops.contract(
        projectors_left[k-1], H[k], '1', '2'), H[k+1], '4', '2'), projectors_right[k+2], '6', '1'),
        [0, 2, 4, 6, 1, 3, 5, 7])
    H_eff_shape = H_effective.tensor.shape
    forward_evolver = tn.Node(
        linalg.expm(-1j * (dt / 2) * H_effective.tensor.reshape([np.prod(H_eff_shape[:4]), np.prod(H_eff_shape[4:])]))
            .reshape([H_eff_shape]))
    M = bops.contract(M, forward_evolver, '0123', '0123')
    [A, C, B, te] = bops.svdTruncation(M, [0, 1], [2, 3], '>*<', maxBondDim=1024)
    if dir == '>>':
        projectors_left[k] = bops.contract(bops.contract(bops.contract(
            projectors_left[k-1], A, '0', '0'), H[k], '02', '20'), A, '02', '01*')
        M = bops.contract(C, B, '1', '0')
        M_ind = k+1
    else:
        projectors_right[k+1] = bops.contract(bops.contract(bops.contract(
            B, projectors_right[k+2], '2', '0'), H[k+1], '12', '03'), B, '21', '12*')
        M = bops.contract(A, C, '2', '0')
        M_ind = k
    H_effective = bops.permute(bops.contract(bops.contract(
        projectors_left[M_ind - 1], H[M_ind], '1', '2'), projectors_right[M_ind + 1], '4', '1'), [0, 2, 4, 1, 3, 5])
    H_eff_shape = H_effective.tensor.shape
    backward_evolver = tn.Node(
        linalg.expm(1j * (dt/2) * H_effective.tensor.reshape([np.prod(H_eff_shape[:3]), np.prod(H_eff_shape[3:])]))\
        .reshape([H_eff_shape]))
    psi[M_ind] = bops.contract(M, backward_evolver, '012', '012')
    return te


def tdvp_sweep(psi: List[tn.Node], H: List[tn.Node],
               projectors_left: List[tn.Node], projectors_right: List[tn.Node], dt):
    max_te = 0
    for k in range(len(psi) - 1):
        te = tdvp_step(psi, H, k, projectors_left, projectors_right, '>>', dt)
        if te > max_te:
            max_te = te
    for k in range(len(psi) - 1, -1, -1):
        te = tdvp_step(psi, H, k, projectors_left, projectors_right, '<<', dt)
        if te > max_te:
            max_te = te
    return te

def get_initial_projectors(psi: List[tn.Node], H: List[tn.Node]):
    res_left = [None for i in range(len(psi))]
    res_right = [None for i in range(len(psi))]
    res_right[-1] = tn.Node(np.eye(psi[-1].shape[2]).reshape([psi[-1].shape[2], 1, psi[-1].shape[2]]))
    for ri in range(len(psi) - 1, 0, -1):
        res_right[ri-1] = bops.contract(bops.contract(bops.contract(
            res_right[ri], psi[ri], '0', '2'), H[ri], '30', '03'), psi[ri], '20', '12*')
        if ri > 0:
            psi = bops.shiftWorkingSite(psi, ri, '<<')
    res_left[0] = tn.Node(np.eye(psi[0].shape[0]).reshape([psi[0].shape[0], 1, psi[0].shape[0]]))
    for ri in range(len(psi) - 1):
        print(ri)
        res_left[ri+1] = bops.contract(bops.contract(bops.contract(
            res_left[ri], psi[ri], '0', '0'), H[ri], '02', '20'), psi[ri], '02', '01*')
        psi = bops.shiftWorkingSite(psi, ri, '>>')
    return res_left, res_right


def get_XXZ_H(n, delta):
    d = 2
    tensor = np.zeros((d, d, 4, 4), dtype=complex)
    tensor[:, :, 0, 0] = np.eye(d)
    tensor[:, :, 0, 1] = basic.pauli2X
    tensor[:, :, 0, 2] = basic.pauli2Y
    tensor[:, :, 0, 3] = basic.pauli2Z * delta
    tensor[:, :, 1, 0] = basic.pauli2X
    tensor[:, :, 2, 0] = basic.pauli2Y
    tensor[:, :, 3, 0] = basic.pauli2Z * delta
    H = [tn.Node(tensor[:, :, 0, :].reshape(list(tensor.shape[:2]) + [1, tensor.shape[3]]))] + \
        [tn.Node(tensor) for i in range(n-2)] + [tn.Node(tensor[:, :, :, 0].reshape(list(tensor.shape[:3]) + [1]))]
    return H


n = 16
delta = 1
H = get_XXZ_H(n, delta)
psi0 = bops.getStartupState(n, d=2)
for k in range(n-1, 0, -1):
    psi0 = bops.shiftWorkingSite(psi0, k, '<<')
for k in range(n-1):
    psi0 = bops.shiftWorkingSite(psi0, k, '>>')
projectors_left, projectors_right = get_initial_projectors(psi0, H)
b = 1