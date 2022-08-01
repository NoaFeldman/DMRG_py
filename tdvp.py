import basicOperations as bops
import numpy as np
import tensornetwork as tn
import pickle
import basicDefs as basic
import sys
from typing import List
import scipy.linalg as linalg
import os
import CBE as cbe


def tensor_overlap(n1, n2):
    return bops.contract(n1, n2, list(range(len(n1.tensor.shape))) + ['*'], list(range(len(n1.tensor.shape)))).tensor * 1


def arnoldi(HL, HR, H, psi, k, num_of_sites=2, max_h_size=50, C=None):
    accuracy = 1e-12
    result = np.zeros((max_h_size, max_h_size), dtype=complex)
    if num_of_sites == 2:
        v = bops.contract(psi[k], psi[k+1], '2', '0')
    elif num_of_sites == 1:
        v = psi[k]
    elif num_of_sites == 0:
        v = C
    max_size = np.prod(v.tensor.shape)
    if max_size < max_h_size:
        max_h_size = max_size
    v.set_tensor(v.get_tensor() / np.sqrt(tensor_overlap(v, v)))
    basis = [bops.copyState([v])[0]]
    size = max_h_size
    for j in range(max_h_size - 1):
        v = applyHToM(HL, HR, H, basis[j], k, num_of_sites=num_of_sites)
        for ri in range(j + 1):
            result[ri, j] = tensor_overlap(basis[ri], v)
            v.set_tensor(v.tensor - result[ri, j] * basis[ri].tensor)
        result[j + 1, j] = np.sqrt(tensor_overlap(v, v))
        if result[j + 1, j] < accuracy or \
                np.any([int(np.abs(tensor_overlap(basis[bi], v) /
                                   tensor_overlap(v, v)) > accuracy)
                        for bi in range(len(basis))]):
            size = j + 1
            break
        basis.append(bops.multNode(bops.copyState([v])[0], 1 / result[j + 1, j]))
    return result[:size, :size], basis

def applyHToM(HL, HR, H, M, k, num_of_sites):
    if num_of_sites == 2:
        return bops.contract(bops.contract(bops.contract(bops.contract(
            HL[k], M, '0', '0'), H[k], '02', '20'), H[k + 1], '41', '20'), HR[k+1], '14', '01')
    elif num_of_sites == 1:
        return bops.contract(bops.contract(bops.contract(
            HL[k], M, '0', '0'), H[k], '02', '20'), HR[k], '13', '01')
    elif num_of_sites == 0:
        return bops.contract(bops.contract(M, HL[k], '0', '0'), HR[k-1], '01', '01')


def apply_time_evolver(arnoldi_T: np.array, arnoldi_basis: List[tn.Node], M: tn.Node, dt):
    time_evolver = linalg.expm(dt * arnoldi_T)
    result = np.zeros(M.tensor.shape, dtype=complex)
    for ri in range(len(time_evolver)):
        m_overlap = tensor_overlap(arnoldi_basis[ri], M)
        for ci in range(len(time_evolver)):
            result += time_evolver[ci, ri] * m_overlap * arnoldi_basis[ci].tensor
    return tn.Node(result)

# k is OC, pair is [k, k+1]
def tdvp_step(psi: List[tn.Node], H: List[tn.Node], k: int,
              HL: List[tn.Node], HR: List[tn.Node], dir: str, dt, max_bond_dim, num_of_sites, is_density_matrix=True):
    if num_of_sites == 1:
        left_ind = k # - int(dir == '>>')
        cbe.get_op_tilde_tr(psi, left_ind, HL, HR, H, dir, D=max_bond_dim)

    T, basis = arnoldi(HL, HR, H, psi, k, num_of_sites)
    if num_of_sites == 2:
        M = bops.contract(psi[k], psi[k+1], '2', '0')
    else:
        M = psi[k]
    M = apply_time_evolver(T, basis, M, dt)

    if dir == '>>':
        if num_of_sites == 2:
            [A, M, te] = bops.svdTruncation(M, [0, 1], [2, 3], '>>', maxBondDim=max_bond_dim)
        elif num_of_sites == 1:
            [A, C, te] = bops.svdTruncation(M, [0, 1], [2], '>>', maxBondDim=max_bond_dim)
        psi[k] = A
        M_ind = k + 1
        if k == len(psi) - num_of_sites:
            # TODO maybe I need to lose C, and add A instead of M for num_of_sites = 1?
            psi[len(psi) - 1] = M
            if is_density_matrix:
                bops.normalize_mps_of_dm(psi, k)
            return te
        HL[k + 1] = bops.contract(bops.contract(bops.contract(
            HL[k], A, '0', '0'), H[k], '02', '20'), A, '02', '01*')
    elif dir == '<<':
        if num_of_sites == 2:
            [M, B, te] = bops.svdTruncation(M, [0, 1], [2, 3], '<<', maxBondDim=max_bond_dim)
        elif num_of_sites == 1:
            [C, B, te] = bops.svdTruncation(M, [0], [1, 2], '<<', maxBondDim=max_bond_dim)
        B_ind = k + num_of_sites - 1
        psi[B_ind] = B
        M_ind = k
        if k == 0:
            # TODO maybe I need to lose C, and add B instead of M for num_of_sites = 1?
            psi[0] = M
            if is_density_matrix:
                bops.normalize_mps_of_dm(psi, k)
            return te
        HR[B_ind - 1] = bops.contract(bops.contract(bops.contract(
            B, HR[B_ind], '2', '0'), H[B_ind], '12', '03'), B, '21', '12*')
    if len(te) > 0: print(max(te))

    if num_of_sites == 2:
        psi[M_ind] = bops.copyState([M])[0]
        T, basis = arnoldi(HL, HR, H, psi, M_ind, num_of_sites=1)
        M = apply_time_evolver(T, basis, M, - dt)
        psi[M_ind] = M
    elif num_of_sites == 1:
        T, basis = arnoldi(HL, HR, H, psi, M_ind, num_of_sites=0, C=C) # TODO
        C = apply_time_evolver(T, basis, C, - dt)
        if dir == '>>':
            psi[k+1] = bops.contract(C, psi[k+1], '1', '0')
        elif dir == '<<':
            psi[k - 1] = bops.contract(psi[k - 1], C, '2', '0')
    if is_density_matrix:
        bops.normalize_mps_of_dm(psi, M_ind - 1)
    return te


def tdvp_sweep(psi: List[tn.Node], H: List[tn.Node],
               projectors_left: List[tn.Node], projectors_right: List[tn.Node], dt, max_bond_dim, num_of_sites):
    max_te = 0
    for k in range(len(psi) - num_of_sites, -1, -1):
        te = tdvp_step(psi, H, k, projectors_left, projectors_right, '<<', dt, max_bond_dim, num_of_sites=num_of_sites)
        if len(te) > 0 and np.max(te) > max_te:
            max_te = te[0]
    for k in range(len(psi) - num_of_sites + 1):
        te = tdvp_step(psi, H, k, projectors_left, projectors_right, '>>', dt, max_bond_dim, num_of_sites=num_of_sites)
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
def vectorized_lindbladian(n, H_terms: List[np.array], L_term: np.array, d=2):
    basic_ops_tensors = [np.kron(np.eye(d ** (ti + 1)), H_terms[ti]) -
                         np.kron(H_terms[ti].transpose(), np.eye(d**(ti + 1))) for ti in range(len(H_terms))]
    basic_ops_tensors[0] += np.kron(L_term.conj(), L_term) \
                            - 0.5 * np.kron(np.eye(d), np.matmul(L_term.conj().T, L_term)) \
                            - 0.5 * np.kron(np.matmul(L_term.conj().T, L_term).T, np.eye(d))
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
            if len(te) > 0: print(max(te))
            single_site_ops.append(bops.permute(op, [1, 2, 0, 3]))
    dim = 2 + np.sum([op.tensor.shape[2] for op in single_site_ops]) - len(H_terms)
    left_tensor = np.zeros((d**2, d**2, 1, dim), dtype=complex)
    mid_tensor = np.zeros((d**2, d**2, dim, dim), dtype=complex)
    right_tensor = np.zeros((d**2, d**2, dim, 1), dtype=complex)
    left_tensor[:, :, 0, -1] = single_site_ops[0].tensor.reshape([d ** 2, d ** 2])
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


def get_gamma_matrix(N, Gamma, nn_num, k, theta):
    result = np.diag(np.ones(N) * Gamma)
    for ni in range(1, nn_num + 1):
        Delta, gamma = get_gnm(ni + 1, Gamma, k, theta)
        for i in range(N - ni):
            result[i, i + ni] = gamma
            result[i + ni, i] = gamma
    return result


def tn_dm_to_matrix(rho, d=2):
    return bops.getExplicitVec(rho, d**2).reshape([d] * 2 * len(rho)).\
        transpose([i * 2 for i in range(len(rho))] + [i * 2 + 1 for i in range(len(rho))]).reshape([d**N, d**N])


