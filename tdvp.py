import basicOperations as bops
import numpy as np
import DMRG as dmrg
import tensornetwork as tn
import pickle
import basicDefs as basic
import sys
from typing import List
import scipy.linalg as linalg


def arnoldi(HL, HR, H, psi, k, max_h_size=50):
    max_size = HL[k].tensor.shape[0] * H[k].tensor.shape[0] * H[k + 1].tensor.shape[0] * HR[k+1].tensor.shape[0]
    if max_size < max_h_size:
        max_h_size = max_size
    accuracy = 1e-9
    result = np.zeros((max_h_size, max_h_size), dtype=complex)
    v = bops.contract(psi[k], psi[k+1], '2', '0')
    v.set_tensor(v.get_tensor() / bops.getNodeNorm(v))
    basis = [bops.copyState([v])[0]]
    size = max_h_size
    for j in range(max_h_size - 1):
        v = applyHToM(HL, HR, H, basis[j], k)
        for ri in range(j + 1):
            result[ri, j] = bops.contract(basis[ri], v, '0123*', '0123').tensor * 1
            v.set_tensor(v.tensor - result[ri, j] * basis[ri].tensor)
        result[j + 1, j] = bops.getNodeNorm(v)
        if result[j + 1, j] < accuracy:
            size = j + 1
            break
        basis.append(bops.multNode(bops.copyState([v])[0], 1 / result[j + 1, j]))
    return result[:size, :size], basis

def applyHToM(HL, HR, H, M, k):
    return bops.contract(bops.contract(bops.contract(bops.contract(
        HL[k], M, '0', '0'), H[k], '02', '20'), H[k + 1], '41', '20'), HR[k+1], '14', '01')
    # return bops.contract(bops.contract(bops.contract(bops.contract(
    #     HL[k], M, '2', '0'), H[k], '12', '21'), H[k+1], '41', '21'), HR[k+1], '14', '21')


def apply_time_evolver(arnoldi_T: np.array, arnoldi_basis: List[tn.Node], M: tn.Node, dt):
    time_evolver = linalg.expm(-1j * dt * arnoldi_T)
    result = np.zeros(M.tensor.shape, dtype=complex)
    for ri in range(len(time_evolver)):
        m_overlap = bops.contract(arnoldi_basis[ri], M, '0123*', '0123').tensor
        for ci in range(len(time_evolver)):
            result += time_evolver[ci, ri] * m_overlap * arnoldi_basis[ci].tensor
    return tn.Node(result)

# k is OC, pair is [k, k+1]
def tdvp_step(psi: List[tn.Node], H: List[tn.Node], k: int,
              HL: List[tn.Node], HR: List[tn.Node], dir: str, dt, max_bond_dim):
    T, basis = arnoldi(HL, HR, H, psi, k)
    M = bops.contract(psi[k], psi[k+1], '2', '0')

    # H_effective = bops.permute(bops.contract(bops.contract(bops.contract(
    #     HL[k], H[k], '1', '2'), H[k+1], '4', '2'), HR[k+1], '6', '1'),
    #     [0, 2, 4, 6, 1, 3, 5, 7])
    # H_eff_shape = H_effective.tensor.shape
    # h = H_effective.tensor.reshape([np.prod(H_eff_shape[:4]), np.prod(H_eff_shape[4:])])
    # forward_evolver = tn.Node(
    #     linalg.expm(-1j * (dt / 2) * H_effective.tensor.reshape([np.prod(H_eff_shape[:4]), np.prod(H_eff_shape[4:])]))
    #         .reshape(list(H_eff_shape)))
    # fe = forward_evolver.tensor.reshape([np.prod(H_eff_shape[:4]), np.prod(H_eff_shape[4:])])
    # v = M.tensor.reshape(np.prod(H_eff_shape[:4]))
    # hM = bops.contract(bops.contract(psi[k], psi[k+1], '2', '0'), H_effective, '0123', '0123')
    # m = bops.contract(bops.contract(psi[k], psi[k+1], '2', '0'), forward_evolver, '0123', '0123')
    #
    # test_t = np.zeros(T.shape, dtype=complex)
    # for i in range(len(basis)):
    #     for j in range(len(basis)):
    #         test_t[i, j] = bops.contract(basis[i], bops.contract(basis[j], H_effective, '0123', '0123'),
    #                                          '0123*', '0123').tensor
    #
    # t_exp_test = np.zeros(T.shape, dtype=complex)
    # time_evolver = linalg.expm(-1j * dt / 2 * T.conj())
    # for i in range(len(basis)):
    #     for j in range(len(basis)):
    #         t_exp_test[i, j] = bops.contract(basis[i], bops.contract(basis[j], forward_evolver, '0123', '0123'),
    #                                          '0123*', '0123').tensor
    # M_basis = np.zeros(len(basis), dtype=complex)
    # m_basis = np.zeros(len(basis), dtype=complex)
    # for bi in range(len(basis)):
    #     M_basis[bi] = bops.contract(basis[bi], M, '0123*', '0123').tensor
    #     m_basis[bi] = bops.contract(basis[bi], m, '0123*', '0123').tensor
    #
    # result = np.zeros(M.tensor.shape, dtype=complex)
    # for ri in range(len(time_evolver)):
    #     m_overlap = bops.contract(basis[ri], M, '0123*', '0123').tensor
    #     for ci in range(len(time_evolver)):
    #         result += T[ci, ri] * m_overlap * basis[ci].tensor

    M = apply_time_evolver(T, basis, M, dt / 2)
    [A, C, B, te] = bops.svdTruncation(M, [0, 1], [2, 3], '>*<', maxBondDim=max_bond_dim)
    if len(te) > 0: print(max(te))
    if dir == '>>':
        projectors_left[k+1] = bops.contract(bops.contract(bops.contract(
            HL[k], A, '0', '0'), H[k], '02', '20'), A, '02', '01*')
        M = bops.contract(C, B, '1', '0')
        psi[k] = A
        M_ind = k + 1
        if k == len(psi) - 2:
            psi[M_ind] = M
            return te
    else:
        projectors_right[k] = bops.contract(bops.contract(bops.contract(
            B, HR[k+1], '2', '0'), H[k+1], '12', '03'), B, '21', '12*')
        M = bops.contract(A, C, '2', '0')
        psi[k+1] = B
        M_ind = k
        if k == 0:
            psi[M_ind] = M
            return te
    H_effective = bops.permute(bops.contract(bops.contract(
        HL[M_ind], H[M_ind], '1', '2'), HR[M_ind], '4', '1'), [0, 2, 4, 1, 3, 5])
    H_eff_shape = H_effective.tensor.shape
    backward_evolver = tn.Node(
        linalg.expm(1j * (dt/2) * H_effective.tensor.reshape([np.prod(H_eff_shape[:3]), np.prod(H_eff_shape[3:])]))\
        .reshape(list(H_eff_shape)))
    psi[M_ind] = bops.contract(M, backward_evolver, '012', '012')
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
    return np.real(g), np.imag(g)


def get_photon_green_L(n, Omega, Gamma, k, theta, opt='NN', nearest_neighbors_num=1, exp_coeffs=[0]):
    d = 2
    sigma = np.array([[0, 1], [0, 0]])
    A = np.kron(np.eye(d), sigma.T)
    B = np.kron(np.eye(d), sigma)
    C = np.kron(sigma.T, np.eye(d))
    D = np.kron(sigma, np.eye(d))
    S = -1j * Omega * (np.kron(np.eye(d), sigma + sigma.T) - np.kron(sigma + sigma.T, np.eye(d)))
    Deltas = np.zeros(nearest_neighbors_num)
    gammas = np.zeros(nearest_neighbors_num)
    for ni in range(nearest_neighbors_num):
        Delta, gamma = get_gnm(ni + 1, Gamma, k, theta)
        Deltas[ni] = Delta
        gammas[ni] = gamma
    pairs = [[[(-1j * Deltas[i] - gammas[i] / 2) * A, B],
              [(1j * Deltas[i] - gammas[i] / 2) * C, D],
              [(-1j * Deltas[i] - gammas[i] / 2) * B + gammas[i] * C, A],
              [(1j * Deltas[i] - gammas[i] / 2) * D + gammas[i] * A, C]] for i in range(nearest_neighbors_num)]
    if opt == 'NN':
        left_tensor = np.zeros((d**2, d**2, 1, 2 + nearest_neighbors_num * 4), dtype=complex)
        left_tensor[:, :, 0, 0] = S
        curr_ind = 1
        for term_i in range(len(pairs[0])):
            for ri in range(nearest_neighbors_num):
                left_tensor[:, :, 0, curr_ind] = pairs[ri][term_i][0]
                curr_ind += 1
        left_tensor[:, :, 0, curr_ind] = np.eye(d**2)

        mid_tensor = np.zeros((d**2, d**2, 2 + nearest_neighbors_num * 4, 2 + nearest_neighbors_num * 4), dtype=complex)
        mid_tensor[:, :, 0, 0] = np.eye(d**2)
        curr_row_ind = 1
        for term_i in range(len(pairs[0])):
            mid_tensor[:, :, curr_row_ind, 0] = pairs[0][term_i][1]
            curr_row_ind += 1
            for ni in range(1, nearest_neighbors_num):
                mid_tensor[:, :, curr_row_ind, curr_row_ind - 1] = np.eye(d**2)
                curr_row_ind += 1
        mid_tensor[:, :, -1, :] = left_tensor[:, :, 0, :]

        right_tensor = np.zeros((d**2, d**2, 2 + nearest_neighbors_num * 4, 1), dtype=complex)
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


test = False
if test:
    N = 6
    d = 2
    L = get_photon_green_L(N, 1, 1, 2 * np.pi / 10, 0, opt='NN', nearest_neighbors_num=2)
    psi0 = [tn.Node(np.array([1, 0, 0, 0]).reshape([1, d**2, 1])) for n in range(N)]
    psi = bops.copyState(psi0)
    projectors_left, projectors_right = get_initial_projectors(psi, L)
    rho_explicit_0 = np.zeros(d**(2*N), dtype=complex)
    rho_explicit_0[0] = 1
    rho_explicit = np.copy(rho_explicit_0)
    curr = L[0]
    for i in range(1, N):
        curr = bops.contract(curr, L[i], [i * 2 + 1], '2')
    L_explicit = curr.tensor\
        .transpose([2, 0, 3, 5, 7, 9, 11, 1, 4, 6, 8, 10, 12, 13])\
        .reshape([d**(2 * N), d**(2*N)])
    dt = 1e-2
    time_evolver_explicit = linalg.expm(-1j * dt * L_explicit)
    timesteps = 10
    overlaps_tdvp = np.zeros(timesteps)
    overlaps_exact = np.zeros(timesteps)
    for ti in range(timesteps):
        tdvp_sweep(psi, L, projectors_left, projectors_right, dt, max_bond_dim=128)
        rho_explicit = np.matmul(time_evolver_explicit, rho_explicit)
        overlaps_tdvp[ti] = np.abs(bops.getOverlap(psi0, psi))
        overlaps_exact[ti] = np.abs(np.matmul(rho_explicit, rho_explicit_0))
        print(ti)
    import matplotlib.pyplot as plt
    plt.plot(overlaps_tdvp, color='black')
    plt.plot(overlaps_exact, '--', color='orange')
    plt.show()


N = int(sys.argv[1])
k = 2 * np.pi / 10
theta = 0
nn_num = int(sys.argv[2])
Gamma = 1
Omega = float(sys.argv[3]) / Gamma 
d = 2
outdir = sys.argv[4]
L = get_photon_green_L(N, Omega, Gamma, k, theta, 'NN', nn_num)
psi = [tn.Node(np.array([0, 0, 0, 1]).reshape([1, d**2, 1])) for n in range(N)]
projectors_left, projectors_right = get_initial_projectors(psi, L)
dt = 1e-2
timesteps = int(sys.argv[5])
Z = np.array([1, 0, 0, -1]).reshape([1, d**2, 1])
I = np.eye(2).reshape([1, d**2, 1])
z_expect = np.zeros(timesteps, dtype=complex)
bond_dims = np.zeros(timesteps, dtype=complex)
for ti in range(timesteps):
    print(ti)
    tdvp_sweep(psi, L, projectors_left, projectors_right, dt, 1024)
    for si in range(N):
        z_expect[ti] += bops.getOverlap(psi,
                            [tn.Node(I) for i in range(si)] + [tn.Node(Z)] + [tn.Node(I) for i in range(si + 1, N)])
    bond_dims[ti] = psi[int(len(psi)/2)].tensor.shape[0]
    if ti % 10 == 0:
        with open(outdir + '/tdvp_N_' + str(N) + '_Omega_' + str(Omega) + '_nn_' + str(nn_num), 'wb') as f:
            pickle.dump([z_expect, bond_dims], f)
        with open(outdir + '/mid_state_N_' + str(N) + '_Omega_' + str(Omega) + '_nn_' + str(nn_num), 'wb') as f:
            pickle.dump([ti, psi], f)
if outdir == 'results':
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(list(range(timesteps)), z_expect)
    axs[1].plot(bond_dims)
    plt.show()

