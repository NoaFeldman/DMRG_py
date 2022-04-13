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
        projectors_left[k], H[k], '1', '2'), H[k+1], '4', '2'), projectors_right[k+1], '6', '1'),
        [0, 2, 4, 6, 1, 3, 5, 7])
    H_eff_shape = H_effective.tensor.shape
    forward_evolver = tn.Node(
        linalg.expm(-1j * (dt / 2) * H_effective.tensor.reshape([np.prod(H_eff_shape[:4]), np.prod(H_eff_shape[4:])]))
            .reshape(list(H_eff_shape)))
    M = bops.contract(M, forward_evolver, '0123', '0123')
    [A, C, B, te] = bops.svdTruncation(M, [0, 1], [2, 3], '>*<', maxBondDim=1024)
    if dir == '>>':
        projectors_left[k+1] = bops.contract(bops.contract(bops.contract(
            projectors_left[k], A, '0', '0'), H[k], '02', '20'), A, '02', '01*')
        M = bops.contract(C, B, '1', '0')
        psi[k] = A
        M_ind = k + 1
    else:
        projectors_right[k] = bops.contract(bops.contract(bops.contract(
            B, projectors_right[k+1], '2', '0'), H[k+1], '12', '03'), B, '21', '12*')
        M = bops.contract(A, C, '2', '0')
        psi[k+1] = B
        M_ind = k
    H_effective = bops.permute(bops.contract(bops.contract(
        projectors_left[M_ind], H[M_ind], '1', '2'), projectors_right[M_ind], '4', '1'), [0, 2, 4, 1, 3, 5])
    H_eff_shape = H_effective.tensor.shape
    backward_evolver = tn.Node(
        linalg.expm(1j * (dt/2) * H_effective.tensor.reshape([np.prod(H_eff_shape[:3]), np.prod(H_eff_shape[3:])]))\
        .reshape(list(H_eff_shape)))
    psi[M_ind] = bops.contract(M, backward_evolver, '012', '012')
    return te


def tdvp_sweep(psi: List[tn.Node], H: List[tn.Node],
               projectors_left: List[tn.Node], projectors_right: List[tn.Node], dt):
    max_te = 0
    for k in range(len(psi) - 2, -1, -1):
        te = tdvp_step(psi, H, k, projectors_left, projectors_right, '<<', dt)
        if len(te) > 0 and np.max(te) > max_te:
            max_te = te[0]
    for k in range(len(psi) - 1):
        te = tdvp_step(psi, H, k, projectors_left, projectors_right, '>>', dt)
        if len(te) > 0 and np.max(te) > max_te:
            max_te = te[0]
    return max_te

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
        res_left[ri+1] = bops.contract(bops.contract(bops.contract(
            res_left[ri], psi[ri], '0', '0'), H[ri], '02', '20'), psi[ri], '02', '01*')
        psi = bops.shiftWorkingSite(psi, ri, '>>')
    return res_left, res_right


def get_XXZ_H(n, delta):
    d = 2
    tensor = np.zeros((d, d, 5, 5), dtype=complex)
    tensor[:, :, 0, 0] = np.eye(d)
    tensor[:, :, 1, 1] = np.eye(d)
    tensor[:, :, 0, 2] = basic.pauli2X
    tensor[:, :, 2, 1] = basic.pauli2X
    tensor[:, :, 0, 3] = basic.pauli2Y
    tensor[:, :, 3, 1] = basic.pauli2Y
    tensor[:, :, 0, 4] = basic.pauli2Z * delta
    tensor[:, :, 4, 1] = basic.pauli2Z
    left_tensor = tn.Node(tensor[:, :, 0, :].reshape(list(tensor.shape[:2]) + [1, tensor.shape[3]]))
    right_tensor = np.zeros((d, d, 5, 1), dtype=complex)
    right_tensor[:, :, 1, 0] = np.eye(d)
    right_tensor[:, :, 2, 0] = basic.pauli2X
    right_tensor[:, :, 3, 0] = basic.pauli2Y
    right_tensor[:, :, 4, 0] = basic.pauli2Z
    H = [tn.Node(left_tensor)] + [tn.Node(tensor) for i in range(n-2)] + [tn.Node(right_tensor)]
    return H


n = 8
delta = 1
H = get_XXZ_H(n, delta)
psi0 = bops.getStartupState(int(n/2), d=2)
psi0, E0, truncErrs = dmrg.DMRG(psi0, [np.eye(2) * 0 for i in range(n)],
    [np.kron(basic.pauli2Z, basic.pauli2Z) * 0.5  + np.kron(basic.pauli2X, basic.pauli2X) + np.kron(basic.pauli2Y, basic.pauli2Y) for i in range(n-1)],
    d=2, maxBondDim=1024, accuracy=1e-12, initial_bond_dim=2)
psi0 = bops.copyState(psi0) + bops.copyState(psi0)
# psi0 = [tn.Node(np.array([[[1], [0]]])) for i in range(int(n/2))] + [tn.Node(np.array([[[0], [1]]])) for i in range(int(n/2))]
for k in range(n-1, 0, -1):
    psi0 = bops.shiftWorkingSite(psi0, k, '<<')
for k in range(n-1):
    psi0 = bops.shiftWorkingSite(psi0, k, '>>')
projectors_left, projectors_right = get_initial_projectors(psi0, H)
psi = bops.copyState(psi0)

psi0_vec = bops.getExplicitVec(psi0, d=2)

curr = bops.contract(H[0], H[1], '3', '2')
for i in range(2, n):
    curr = bops.contract(curr, H[i], [1 + 2 * i], '2')
H_mat = bops.permute(curr, [2, 0, 3, 5, 7, 9, 11, 13, 15, 1, 4, 6, 8, 10, 12, 14, 16, 17]).tensor.reshape([2**8, 2**8])

dt = delta * 1e-2
timesteps = 100

time_evolver = linalg.expm(H_mat * 1j * dt)
psi_vec = np.copy(psi0_vec)
Xs = np.eye(1)
for i in range(2):
    Xs = np.kron(Xs, basic.pauli2X)
for i in range(2, n):
    Xs = np.kron(Xs, np.eye(2))
xs_ops = [tn.Node(basic.pauli2X) for i in range(2)] + [tn.Node(np.eye(2)) for i in range(2, n)]

p2s_tdvp = np.zeros(timesteps, dtype=complex)
p2s_exact = np.zeros(timesteps, dtype=complex)
overlaps_tdvp = np.zeros(timesteps, dtype=complex)
overlaps_exact = np.zeros(timesteps, dtype=complex)
overlaps_test = np.zeros(timesteps, dtype=complex)
for ti in range(timesteps):
    if ti == 30:
        b = 1
    te = tdvp_sweep(psi, H, projectors_left, projectors_right, dt)
    print(ti, te)
    psi_vec = np.matmul(time_evolver, psi_vec)
    overlaps_tdvp[ti] = bops.getOverlap(psi0, psi)
    overlaps_exact[ti] = np.matmul(np.conj(psi0_vec.T), psi_vec)
    p2s_tdvp[ti] = bops.getRenyiEntropy(psi, 2, int(n/2))
    rho = np.outer(psi_vec, np.conj(psi_vec.T)).reshape([2**(int(n/2)), 2**(int(n/2)), 2**(int(n/2)), 2**(int(n/2))])
    rho_A = np.tensordot(rho, np.eye(2**(int(n/2))), [[1, 3], [0, 1]])
    p2s_exact[ti] = np.trace(np.matmul(rho_A, rho_A))
plt.plot(np.abs(overlaps_tdvp), color='red')
plt.plot(np.abs(overlaps_exact), '--k', color='black')
plt.plot(np.abs(p2s_tdvp), color='blue')
plt.plot(np.abs(p2s_exact), '--k', color='orange')
plt.show()
b = 1