import basicOperations as bops
import numpy as np
import DMRG as dmrg
import tensornetwork as tn
import pickle
import basicDefs as basic
import sys
from typing import List
import scipy.linalg as linalg


def tensor_overlap(n1, n2, num_of_sites=2):
    if num_of_sites == 2:
        return bops.contract(n1, n2, '0123*', '0123').tensor * 1
    else:
        return bops.contract(n1, n2, '012*', '012').tensor * 1


def arnoldi(HL, HR, H, psi, k, num_of_sites=2, max_h_size=50):
    max_size = HL[k].tensor.shape[0] * H[k].tensor.shape[0] * H[k + 1].tensor.shape[0] * HR[k+1].tensor.shape[0]
    if max_size < max_h_size:
        max_h_size = max_size
    accuracy = 1e-12
    result = np.zeros((max_h_size, max_h_size), dtype=complex)
    if num_of_sites == 2:
        v = bops.contract(psi[k], psi[k+1], '2', '0')
    else:
        v = psi[k]
        # h = bops.contract(bops.contract(HL[k], H[k], '1', '2'), HR[k], '4', '1').tensor\
        #     .transpose([1, 3, 5, 0, 2, 4]).reshape(
        #     [HL[k].tensor.shape[0] * 4 * HR[k].tensor.shape[0], HL[k].tensor.shape[0] * 4 * HR[k].tensor.shape[0]])
    v.set_tensor(v.get_tensor() / np.sqrt(tensor_overlap(v, v, num_of_sites=num_of_sites)))
    basis = [bops.copyState([v])[0]]
    size = max_h_size
    for j in range(max_h_size - 1):
        v = applyHToM(HL, HR, H, basis[j], k, num_of_sites=num_of_sites)
        for ri in range(j + 1):
            result[ri, j] = tensor_overlap(basis[ri], v, num_of_sites)
            v.set_tensor(v.tensor - result[ri, j] * basis[ri].tensor)
        result[j + 1, j] = np.sqrt(tensor_overlap(v, v, num_of_sites=num_of_sites))
        if result[j + 1, j] < accuracy or \
                np.any([int(np.abs(tensor_overlap(basis[bi], v, num_of_sites) /
                                   tensor_overlap(v, v, num_of_sites=num_of_sites)) > accuracy)
                        for bi in range(len(basis))]):
            size = j + 1
            break
        basis.append(bops.multNode(bops.copyState([v])[0], 1 / result[j + 1, j]))
    return result[:size, :size], basis

def applyHToM(HL, HR, H, M, k, num_of_sites):
    if num_of_sites == 2:
        return bops.contract(bops.contract(bops.contract(bops.contract(
            HL[k], M, '0', '0'), H[k], '02', '20'), H[k + 1], '41', '20'), HR[k+1], '14', '01')
    else:
        return bops.contract(bops.contract(bops.contract(
            HL[k], M, '0', '0'), H[k], '02', '20'), HR[k], '13', '01')


def apply_time_evolver(arnoldi_T: np.array, arnoldi_basis: List[tn.Node], M: tn.Node, dt, num_of_sites=2):
    time_evolver = linalg.expm(dt * arnoldi_T)
    result = np.zeros(M.tensor.shape, dtype=complex)
    for ri in range(len(time_evolver)):
        if num_of_sites == 2:
            m_overlap = bops.contract(arnoldi_basis[ri], M, '0123*', '0123').tensor
        else:
            m_overlap = bops.contract(arnoldi_basis[ri], M, '012*', '012').tensor
        for ci in range(len(time_evolver)):
            result += time_evolver[ci, ri] * m_overlap * arnoldi_basis[ci].tensor
    return tn.Node(result)

# k is OC, pair is [k, k+1]
def tdvp_step(psi: List[tn.Node], H: List[tn.Node], k: int,
              HL: List[tn.Node], HR: List[tn.Node], dir: str, dt, max_bond_dim):
    T, basis = arnoldi(HL, HR, H, psi, k)
    M = bops.contract(psi[k], psi[k+1], '2', '0')

    H_effective = bops.contract(bops.contract(bops.contract(HL[k], H[k], '1', '2'), H[k + 1], '4', '2'), HR[k+1], '6', '1')
    H_effective_mat = H_effective.tensor.transpose([1, 3, 5, 7, 0, 2, 4, 6]).\
        reshape([int(np.sqrt(np.prod(H_effective.tensor.shape))), int(np.sqrt(np.prod(H_effective.tensor.shape)))])
    m_vec = M.tensor.reshape([np.prod(M.tensor.shape)])
    hv = bops.contract(H_effective, M, '0246', '0123').tensor.reshape([np.prod(M.tensor.shape)])
    hv_exact = np.matmul(H_effective_mat, m_vec)
    h_evolved = np.matmul(linalg.expm(dt * H_effective_mat / 2), m_vec)
    M = apply_time_evolver(T, basis, M, dt)
    m_evolved = M.tensor.reshape([np.prod(M.tensor.shape)])

    [A, C, B, te] = bops.svdTruncation(M, [0, 1], [2, 3], '>*<', maxBondDim=max_bond_dim)
    # print(C.tensor.shape)
    if len(te) > 0: print(max(te))
    if dir == '>>':
        M = bops.contract(C, B, '1', '0')
        psi[k] = A
        M_ind = k + 1
        if k == len(psi) - 2:
            psi[M_ind] = M
            return te
        HL[k + 1] = bops.contract(bops.contract(bops.contract(
            HL[k], A, '0', '0'), H[k], '02', '20'), A, '02', '01*')
    else:
        M = bops.contract(A, C, '2', '0')
        psi[k+1] = B
        M_ind = k
        if k == 0:
            psi[M_ind] = M
            return te
        HR[k] = bops.contract(bops.contract(bops.contract(
            B, HR[k + 1], '2', '0'), H[k + 1], '12', '03'), B, '21', '12*')

    psi[M_ind] = bops.copyState([M])[0]
    T, basis = arnoldi(HL, HR, H, psi, M_ind, num_of_sites=1)

    # H_effective = bops.permute(bops.contract(bops.contract(
    #     HL[M_ind], H[M_ind], '1', '2'), HR[M_ind], '4', '1'), [1, 3, 5, 0, 2, 4])
    # H_effective_mat = H_effective.tensor.\
    #     reshape([int(np.sqrt(np.prod(H_effective.tensor.shape))), int(np.sqrt(np.prod(H_effective.tensor.shape)))])
    # m_vec = M.tensor.reshape([np.prod(M.tensor.shape)])
    # hv = bops.contract(H_effective, M, '345', '012').tensor.reshape([np.prod(M.tensor.shape)])
    # hv_exact = np.matmul(H_effective_mat, m_vec)
    # h_evolved = np.matmul(linalg.expm(-dt * H_effective_mat / 2), m_vec)
    M = apply_time_evolver(T, basis, M, - dt, num_of_sites=1)
    # m_evolved = M.tensor.reshape([np.prod(M.tensor.shape)])
    # # print(int(np.log(max(np.abs(m_evolved - h_evolved))) / np.log(10)))

    psi[M_ind] = M
    return te


def tdvp_sweep(psi: List[tn.Node], H: List[tn.Node],
               projectors_left: List[tn.Node], projectors_right: List[tn.Node], dt, max_bond_dim):
    max_te = 0
    for k in range(len(psi) - 2, -1, -1):
        te = tdvp_step(psi, H, k, projectors_left, projectors_right, '<<', dt, max_bond_dim)
        if len(te) > 0 and np.max(te) > max_te:
            max_te = te[0]
    for k in range(len(psi) - 1):
        te = tdvp_step(psi, H, k, projectors_left, projectors_right, '>>', dt, max_bond_dim)
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


def get_gnm(r, gamma, k, theta):
    kr = k * r
    g = -gamma * 3 / 4 * np.exp(1j * kr) / kr * (1 + (1j * kr - 1) / kr**2 + (-1 + 3 * (1 - 1j * kr) / kr**2) * np.cos(theta)**2)
    return np.real(g), -2 * np.imag(g)


def get_photon_green_L(n, Omega, Gamma, k, theta, sigma, opt='NN', case='kernel', nearest_neighbors_num=1, exp_coeffs=[0]):
    d = 2
    A = np.kron(np.eye(d), sigma.T)
    B = np.kron(np.eye(d), sigma)
    C = np.kron(sigma.T, np.eye(d))
    D = np.kron(sigma, np.eye(d))
    S = -1j * Omega * (np.kron(np.eye(d), sigma + sigma.T) - np.kron(sigma + sigma.T, np.eye(d))) \
        + Gamma * (np.kron(sigma, sigma)
                   - 0.5 * (np.kron(np.matmul(sigma.T, sigma), np.eye(d)) + np.kron(np.eye(d), np.matmul(sigma.T, sigma)))) \
        + 1j * (np.kron(np.matmul(sigma.T, sigma), np.eye(d)) - np.kron(np.eye(d), np.matmul(sigma.T, sigma)))

    Deltas = np.zeros(nearest_neighbors_num)
    gammas = np.zeros(nearest_neighbors_num)
    for ni in range(nearest_neighbors_num):
        Delta, gamma = get_gnm(ni + 1, Gamma, k, theta)
        if case == 'kernel':
            Deltas[ni] = Delta
            gammas[ni] = gamma
        elif case == 'dicke':
            Deltas[ni] = 1
            gammas[ni] = 1
    pairs = [[[(-1j * Deltas[i] - gammas[i] / 2) * A + gammas[i] * D for i in range(nearest_neighbors_num)], B],
                 [[(1j * Deltas[i] - gammas[i] / 2) * C + gammas[i] * B for i in range(nearest_neighbors_num)], D],
                 [[(-1j * Deltas[i] - gammas[i] / 2) * B for i in range(nearest_neighbors_num)], A],
                 [[(1j * Deltas[i] - gammas[i] / 2) * D for i in range(nearest_neighbors_num)], C]]
    operators_len = 2 + 4 * nearest_neighbors_num
    if opt == 'NN':
        left_tensor = np.zeros((d**2, d**2, 1, operators_len), dtype=complex)
        left_tensor[:, :, 0, 0] = S.T
        curr_ind = 1
        for pi in range(len(pairs)):
            for ri in range(len(pairs[pi][0])):
                left_tensor[:, :, 0, curr_ind] = pairs[pi][0][ri].T
                curr_ind += 1
        nothing_yet_ind = curr_ind
        left_tensor[:, :, 0, nothing_yet_ind] = np.eye(d**2)

        mid_tensor = np.zeros((d**2, d**2, operators_len, operators_len), dtype=complex)
        mid_tensor[:, :, 0, 0] = np.eye(d**2)
        mid_tensor[:, :, nothing_yet_ind, 0] = S.T
        mid_tensor[:, :, nothing_yet_ind, nothing_yet_ind] = np.eye(d**2)
        curr_ind = 1
        for pi in range(len(pairs)):
            mid_tensor[:, :, curr_ind, 0] = pairs[pi][1].T
            for ri in range(len(pairs[pi][0])):
                mid_tensor[:, :, nothing_yet_ind, curr_ind] = pairs[pi][0][ri].T
                if ri > 0:
                    mid_tensor[:, :, curr_ind, curr_ind - 1] = np.eye(d**2)
                curr_ind += 1

        right_tensor = np.zeros((d**2, d**2, operators_len, 1), dtype=complex)
        right_tensor[:, :, :, 0] = mid_tensor[:, :, :, 0]
    elif opt == 'exp':
        pairs = pairs[0]
        left_tensor = np.zeros((d**2, d**2, 1, 2 + len(pairs)), dtype=complex)
        right_tensor = np.zeros((d**2, d**2, 2 + len(pairs), 1), dtype=complex)
        left_tensor[:, :, 0, 0] = S
        left_tensor[:, :, 0, 0] = np.eye(d**2)
        for pi in range(len(pairs)):
            left_tensor[:, :, 0, pi + 1] = pairs[pi][0]
            right_tensor[:, :, pi + 1, 0] = pairs[pi][1]
        left_tensor[:, :, 0, -1] = np.eye(d**2)
        left_tensor[:, :, -1, 0] = S
        mid_tensor = np.zeros((d**2, d**2, 2 + len(pairs), 2+ len(pairs)))
        mid_tensor[:, :, :, 0] = right_tensor[:, :, :, 0]
        mid_tensor[:, :, -1, :] = left_tensor[:, :, 0, :]
        for pi in range(len(pairs)):
            mid_tensor[:, :, pi + 1, pi + 1] = exp_coeffs[pi] * np.eye(d**2)
    L = [tn.Node(left_tensor)] + [tn.Node(mid_tensor) for i in range(n - 2)] + [tn.Node(right_tensor)]
    I = np.eye(4)
    expected_v = np.kron(S, I) + np.kron(I, S) + 0.5 * np.kron(A, B) - 0.5 * np.kron(C, D) \
                 + 0.5 * np.kron(B, A) + np.kron(C, A) -  0.5 * np.kron(D, C) + np.kron(A, C)
    return L


def get_gamma_matrix(N, Gamma, nn_num, k, theta):
    result = np.diag(np.ones(N) * Gamma)
    for ni in range(1, nn_num + 1):
        Delta, gamma = get_gnm(ni + 1, Gamma, k, theta)
        for i in range(N - ni):
            result[i, i + ni] = gamma
            result[i + ni, i] = gamma
    return result


def tn_dm_to_matrix(rho):
    return bops.getExplicitVec(rho, d**2).reshape([d] * 2 * len(rho)).\
        transpose([i * 2 for i in range(len(rho))] + [i * 2 + 1 for i in range(len(rho))]).reshape([d**N, d**N])


N = int(sys.argv[1])
k = 2 * np.pi / 10
theta = 0
nn_num = int(sys.argv[2])
Gamma = 1
Omega = float(sys.argv[3]) / Gamma
case = sys.argv[4]
d = 2
outdir = sys.argv[5]
dt = 1e-3
sigma = np.array([[0, 0], [1, 0]])
timesteps = int(sys.argv[6])
results_to = sys.argv[7]
if results_to == 'plot':
    import matplotlib.pyplot as plt
L = get_photon_green_L(N, Omega, Gamma, k, theta, sigma, case=case, nearest_neighbors_num=nn_num)
psi = [tn.Node(np.array([1, 0, 0, 0]).reshape([1, d**2, 1])) for n in range(N)]
if N <= 10:
    I = np.eye(2).reshape([1, d ** 2, 1])
    Deltas = np.ones(nn_num + 1)
    gammas = np.zeros(nn_num + 1)
    gammas[0] = Gamma
    for ni in range(nn_num):
        Delta, gamma = get_gnm(ni + 1, Gamma, k, theta)
        Deltas[ni] = 1 # Delta
        gammas[ni + 1] = 1 # gamma
    L_exact = np.zeros((d**(2*N), d**(2*N)), dtype=complex)
    sigmas = []
    for i in range(N):
        sigmas.append(np.kron(np.eye(d**(i)), np.kron(sigma, np.eye(d**(N - i - 1)))))

    for n in range(N):
        for m in range(N):
            if np.abs(m - n) <= nn_num:
                L_exact += (-1j * Deltas[np.abs(m - n)] - 0.5 * gammas[np.abs(m - n)]) * \
                           np.kron(np.eye(d**N), np.matmul(sigmas[n].T, sigmas[m]))
                L_exact += (1j * Deltas[np.abs(m - n)] - 0.5 * gammas[np.abs(m - n)]) * \
                           np.kron(np.matmul(sigmas[n].T, sigmas[m]), np.eye(d**N))
                L_exact += gammas[np.abs(m - n)] * np.kron(sigmas[n], sigmas[m])

    explicit_L = bops.contract(L[0], L[1], '3', '2')
    explicit_rho = bops.contract(psi[0], psi[1], '2', '0')
    for ni in range(2, N):
        explicit_L = bops.contract(explicit_L, L[ni], [2 * ni + 1], '2')
        explicit_rho = bops.contract(explicit_rho, psi[ni], [ni + 1], '0')
    L_mat = explicit_L.tensor.reshape([d] * 4 * N).transpose([4 * i for i in range(N)]
                                                             + [1 + 4 * i for i in range(N)]
                                                             + [2 + 4 * i for i in range(N)]
                                                             + [3 + 4 * i for i in range(N)])\
        .reshape([d**(2 * N), d**(2 * N)])
    rho_vec = bops.getExplicitVec(psi, d**2)
    evolver = linalg.expm(L_mat.T * dt)
    J_expect = np.zeros(timesteps)
    z_inds = [[i + d**N * i,
               2 * bin(i).split('b')[1].count('1') - N]
              for i in range(d**N)]
    J = np.zeros(sigmas[0].shape)
    for i in range(N):
        J += sigmas[i]
    JdJ = np.matmul(J.conj().T, J)
    for ti in range(timesteps):
        J_expect[ti] = np.abs(np.trace(np.matmul(JdJ, rho_vec.reshape([d**N, d**N]))))
        rho_vec = np.matmul(evolver, rho_vec)
    if results_to == 'plot':
        plt.plot(J_expect)
        # plt.show()
    else:
        with open(outdir + '/explicit_J_expect', 'wb') as f:
            pickle.dump(J_expect, f)

projectors_left, projectors_right = get_initial_projectors(psi, L)
I = np.eye(2).reshape([1, d**2, 1])
J_expect = np.zeros(timesteps, dtype=complex)
bond_dims = np.zeros(timesteps, dtype=complex)
for ti in range(timesteps):
    print('---')
    print(ti)
    for si in range(N):
        J_expect[ti] += bops.getOverlap(psi,
                            [tn.Node(I) for i in range(si)] + [tn.Node(np.matmul(sigma.T, sigma).reshape([1, d**2, 1]))]
                                        + [tn.Node(I) for i in range(si + 1, N)])
        for sj in range(N):
            if si != sj:
                J_expect[ti] += bops.getOverlap(psi,
                                [tn.Node(I) for i in range(si)] + [tn.Node(sigma.T.reshape([1, d**2, 1]))]
                                                + [tn.Node(I) for i in range(si + 1, sj)] + [tn.Node(sigma.reshape([1, d**2, 1]))]
                                                + [tn.Node(I) for i in range(sj +1, N)])
    # TODO look at entanglement
    bond_dims[ti] = psi[int(len(psi)/2)].tensor.shape[0]
    rho_explicit = tn_dm_to_matrix(psi)
    print(np.trace(rho_explicit))
    print(min(np.diag(rho_explicit)))
    print(np.amax(np.abs(rho_explicit - rho_explicit.T.conj())))
    tdvp_sweep(psi, L, projectors_left, projectors_right, dt, max_bond_dim=128)
    if True: # ti % 10 == 0:
        with open(outdir + '/tdvp_N_' + str(N) + '_Omega_' + str(Omega) + '_nn_' + str(nn_num), 'wb') as f:
            pickle.dump([J_expect, bond_dims], f)
        with open(outdir + '/mid_state_N_' + str(N) + '_Omega_' + str(Omega) + '_nn_' + str(nn_num) + '_ti_' + str(ti), 'wb') as f:
            pickle.dump([ti, psi], f)
if results_to == 'plot':
    plt.plot(list(range(timesteps)), np.abs(J_expect))
    plt.show()