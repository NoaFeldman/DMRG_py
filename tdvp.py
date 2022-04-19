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


# dm site is
#      2
#      |
#   0--O--3
#      |
#      1
def vectorize_density_matrix(rho):
    return [bops.unifyLegs(node, [1, 2]) for node in rho]


def is_end_of_term(oi):
    tst = 0
    for i in range(2, 100):
        if tst < oi:
            tst += i
        else:
            if tst == oi:
                return True
            else:
                return False


def H_op_transposition(site_num):
    result = []
    for i in range(site_num):
        result = result + [i, site_num + i, 2 * site_num + i, 3 * site_num + i]
    return result


# translational_invariant
# TODO transpose by sites
def vectorized_lindbladian(n, H_terms: List[np.array], d=2):
    basic_ops_tensors = [np.kron(np.eye(d**(ti + 1)), H_terms[ti]) -
                         np.kron(H_terms[ti].transpose(), np.eye(d**(ti + 1))) for ti in range(len(H_terms))]
    basic_ops = [tn.Node(basic_ops_tensors[ti].reshape([d] * 4 * (ti + 1))
                         .transpose(H_op_transposition(ti + 1))
                         .reshape([1] + [d**2] * 2 * (ti+1) + [1]))
                 for ti in range(len(basic_ops_tensors))]
    single_site_ops = []
    for ti in range(len(H_terms)):
        to_decompose = basic_ops[ti]
        for si in range(ti + 1):
            [op, to_decompose, te] = \
                bops.svdTruncation(to_decompose, [0, 1, 2], list(range(3, len(to_decompose.tensor.shape))), '<<')
            single_site_ops.append(bops.permute(op, [1, 2, 0, 3]))
    dim = 1 + np.sum([op.tensor.shape[2] for op in single_site_ops]) - len(H_terms) + 1
    left_tensor = np.zeros((d**2, d**2, 1, dim), dtype=complex)
    mid_tensor = np.zeros((d**2, d**2, dim, dim), dtype=complex)
    right_tensor = np.zeros((d**2, d**2, dim, 1), dtype=complex)
    left_tensor[:, :, 0, -1] = single_site_ops[0].tensor.reshape([d**2, d**2])
    left_tensor[:, :, 0, 0] = np.eye(d**2)
    curr_left_tensor_ind = 1
    curr_op_list_ind = 1
    for ti in range(1, len(H_terms)):
        op = single_site_ops[curr_op_list_ind]
        left_tensor[:, :, 0:1, curr_left_tensor_ind:(curr_left_tensor_ind + op.tensor.shape[3])] = op.tensor
        curr_op_list_ind += 1 + ti
        curr_left_tensor_ind += op.tensor.shape[3]
    mid_tensor[:, :, 0, 0] = np.eye(d**2)
    mid_tensor[:, :, -1, -1] = np.eye(d**2)
    right_tensor[:, :, -1, 0] = np.eye(d**2)
    curr_op_list_ind = 0
    curr_mid_tensor_ind_left = 0
    curr_mid_tensor_ind_right = 1
    for ti in range(len(H_terms)):
        curr_mid_tensor_ind_left = 0
        for si in range(ti):
            op = single_site_ops[curr_op_list_ind + si]
            mid_tensor[:, :, curr_mid_tensor_ind_left:(curr_mid_tensor_ind_left + op.tensor.shape[2]),
                curr_mid_tensor_ind_right:(curr_mid_tensor_ind_right + op.tensor.shape[3])] = op.tensor
            curr_mid_tensor_ind_left = curr_mid_tensor_ind_right
            curr_mid_tensor_ind_right += op.tensor.shape[3]
        op = single_site_ops[curr_op_list_ind + ti]
        mid_tensor[:, :, curr_mid_tensor_ind_left:(curr_mid_tensor_ind_left + op.tensor.shape[2]), dim-1:dim] = op.tensor
        right_tensor[:, :, curr_mid_tensor_ind_left:(curr_mid_tensor_ind_left + op.tensor.shape[2]), 0:1] = op.tensor
        curr_op_list_ind += ti + 1
    return [tn.Node(left_tensor)] + [tn.Node(mid_tensor) for i in range(n - 2)] + [tn.Node(right_tensor)]


def get_XXZ_H_pure(n, delta):
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


def get_XXZ_H(n, delta):
    return vectorized_lindbladian(n, [basic.pauli2Z*1e-12,
                                      np.kron(basic.pauli2X, basic.pauli2X) + np.kron(basic.pauli2Y, basic.pauli2Y) + \
                                      delta * np.kron(basic.pauli2Z, basic.pauli2Z)])

n = 4
delta = 1
H = get_XXZ_H(n, delta)
psi0 = bops.getStartupState(n, d=2)
vec_rho0 = [tn.Node(np.kron(psi0[i].tensor, np.conj(psi0[i].tensor))) for i in range(len(psi0))]
for k in range(n-1, 0, -1):
    vec_rho0 = bops.shiftWorkingSite(vec_rho0, k, '<<')
for k in range(n-1):
    vec_rho0 = bops.shiftWorkingSite(vec_rho0, k, '>>')
projectors_left, projectors_right = get_initial_projectors(vec_rho0, H)
vec_rho = bops.copyState(vec_rho0)

psi0_vec = bops.getExplicitVec(psi0, d=2)

curr = H[0]
for i in range(1, len(H)):
    curr = bops.contract(curr, H[i], [2 * i + 1], '2')
H_mat = np.round(bops.permute(curr, [2, 0, 3, 5, 7, 1, 4, 6, 8, 9]).tensor.reshape([2**8, 2**8]), 6)
h = np.zeros((2**4, 2**4), dtype=complex)
for i in range(n - 1):
    h += np.kron(np.kron(np.eye(2**i), np.kron(basic.pauli2X, basic.pauli2X) + np.kron(basic.pauli2Y, basic.pauli2Y) + \
                                      delta * np.kron(basic.pauli2Z, basic.pauli2Z)), np.eye(2**(n - i - 2)))
dt = delta * 1e-2
timesteps = 1000

time_evolver = linalg.expm(h * 1j * dt)
psi_vec = np.copy(psi0_vec)
Xs = np.eye(1)
for i in range(2):
    Xs = np.kron(Xs, basic.pauli2X)
for i in range(2, n):
    Xs = np.kron(Xs, np.eye(2))
xs_ops = [tn.Node(basic.pauli2X) for i in range(2)] + [tn.Node(np.eye(2)) for i in range(2, n)]

def explicit_dm(vec_rho):
    curr = vec_rho[0]
    for i in range(1, len(vec_rho)):
        curr = bops.contract(curr, vec_rho[i], [i + 1], '0')
    dm = curr.tensor.reshape([2] * len(vec_rho) * 2).transpose([0, 2, 4, 6, 1, 3, 5, 7]).reshape([2**4, 2**4])
    return dm

p2s_tdvp = np.zeros(timesteps, dtype=complex)
p2s_exact = np.zeros(timesteps, dtype=complex)
overlaps_tdvp = np.zeros(timesteps, dtype=complex)
overlaps_exact = np.zeros(timesteps, dtype=complex)
overlaps_test = np.zeros(timesteps, dtype=complex)
dm0 = explicit_dm(vec_rho)
evals, evecs = np.linalg.eigh(dm0)
psi_test0 = evecs[:, -1]
for ti in range(timesteps):
    te = tdvp_sweep(vec_rho, H, projectors_left, projectors_right, dt)
    print(ti, te)
    psi_vec = np.matmul(time_evolver, psi_vec)
    overlaps_tdvp[ti] = bops.getOverlap(vec_rho0, vec_rho)
    overlaps_exact[ti] = np.abs(np.matmul(np.conj(psi0_vec.T), psi_vec))**2
    # dm = explicit_dm(vec_rho)
    # evals, evecs = np.linalg.eigh(dm)
    # overlaps_test[ti] = np.abs(np.matmul(np.conj(psi0_vec.T), evecs[:, -1]))**2
    p2s_tdvp[ti] = 0
    rho = np.outer(psi_vec, np.conj(psi_vec.T)).reshape([2**(int(n/2)), 2**(int(n/2)), 2**(int(n/2)), 2**(int(n/2))])
    rho_A = np.tensordot(rho, np.eye(2**(int(n/2))), [[1, 3], [0, 1]])
    p2s_exact[ti] = np.trace(np.matmul(rho_A, rho_A))
plt.plot(np.array(range(timesteps)), np.abs(overlaps_tdvp), color='red')
plt.plot(1.2 * np.array(range(timesteps)), np.abs(overlaps_exact), '--', color='black')
# plt.plot(np.abs(p2s_tdvp), color='blue')
# plt.plot(np.abs(p2s_exact), '--', color='orange')
plt.show()
b = 1