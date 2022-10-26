import numpy as np
import pickle
import tdvp
import sys
import os
import tensornetwork as tn
import basicOperations as bops
import scipy.linalg as linalg
import swap_dmrg as swap
from typing import List
import time

def get_gnm(gamma, k, theta, nearest_neighbors_num, case):
    if case == 'dicke':
        Deltas = np.ones(nearest_neighbors_num + 1)
        gammas = np.ones(nearest_neighbors_num + 1)
    else:
        Deltas = np.zeros(nearest_neighbors_num)
        gammas = np.zeros(nearest_neighbors_num)
        gammas[0] = gamma
        for ni in range(nearest_neighbors_num):
            r = ni + 1
            kr = k * r
            g = -gamma * 3 / 4 * np.exp(1j * kr) / kr * (1 + (1j * kr - 1) / kr**2 + (-1 + 3 * (1 - 1j * kr) / kr**2) * np.cos(theta)**2)
            Deltas[ni] = np.real(g)
            gammas[ni] = -2 * np.imag(g)
    return Deltas, gammas


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


# https://math.stackexchange.com/questions/1428566/fit-sum-of-exponentials
def fit_exponential(y):
    y = np.array(y)
    S = np.zeros(len(y), dtype=complex)
    SS = np.zeros(len(y), dtype=complex)
    for k in range(1, len(y)):
        S[k] = S[k - 1] + 0.5 * (y[k] + y[k - 1])
        SS[k] = SS[k - 1] + 0.5 * (S[k] + S[k - 1])
    x = np.array(range(len(y))) + 1
    rhs_vec = np.array([np.sum(SS * y),
                        np.sum(S * y),
                        np.sum(x**2 * y),
                        np.sum(x * y),
                        np.sum(y)])
    rhs_mat = np.linalg.inv(
        np.array([[np.sum(SS**2), np.sum(SS*S), np.sum(SS * x**2), np.sum(SS*x), np.sum(SS)],
                        [np.sum(SS*S), np.sum(S**2), np.sum(S * x**2), np.sum(S * x), np.sum(S)],
                        [np.sum(SS * x**2), np.sum(S * x**2), np.sum(x**4), np.sum(x**3), np.sum(x**2)],
                        [np.sum(SS * x), np.sum(S * x), np.sum(x**3), np.sum(x**2), np.sum(x)],
                        [np.sum(SS), np.sum(S), np.sum(x**2), np.sum(x), len(y)]]))
    A_to_E = np.matmul(rhs_mat, rhs_vec)
    p = 0.5 * (A_to_E[1] + np.sqrt(A_to_E[1]**2 + 4 * A_to_E[0]))
    q = 0.5 * (A_to_E[1] - np.sqrt(A_to_E[1]**2 + 4 * A_to_E[0]))
    betas = np.exp(x * p)
    etas = np.exp(x * q)
    rhs_vec = np.array([np.sum(y), np.sum(betas * y), np.sum(etas * y)])
    rhs_mat = np.linalg.inv(
        np.array([[len(y), np.sum(betas), np.sum(etas)],
                  [np.sum(betas), np.sum(betas**2), np.sum(betas * etas)],
                  [np.sum(etas), np.sum(betas * etas), np.sum(etas**2)]]))
    a_b_c = np.matmul(rhs_mat, rhs_vec)
    return a_b_c[0], a_b_c[1], a_b_c[2], p, q


def get_single_L_term(Omega, Gamma, sigma):
    G = -1j * Gamma / 2
    return -1j * (np.kron(np.eye(d), np.conj(Omega) * sigma + Omega * sigma.T) -
                  np.kron(Omega * sigma + np.conj(Omega) * sigma.T, np.eye(d))) \
        + Gamma * np.kron(sigma, sigma) \
        -1j * (G * np.kron(np.eye(d), np.matmul(sigma.T, sigma).T) -
               np.conj(G) * np.kron(np.matmul(sigma.T, sigma), np.eye(d)))


def get_pair_L_terms(Deltas, gammas, nearest_neighbors_num, sigma):
    A = np.kron(np.eye(d), sigma.T)
    B = np.kron(np.eye(d), sigma)
    C = np.kron(sigma.T, np.eye(d))
    D = np.kron(sigma, np.eye(d))
    return [[[(-1j * Deltas[i] - gammas[i] / 2) * A + gammas[i] * D for i in range(nearest_neighbors_num)], B],
     [[(1j * Deltas[i] - gammas[i] / 2) * C + gammas[i] * B for i in range(nearest_neighbors_num)], D],
     [[(-1j * Deltas[i] - gammas[i] / 2) * B for i in range(nearest_neighbors_num)], A],
     [[(1j * Deltas[i] - gammas[i] / 2) * D for i in range(nearest_neighbors_num)], C]]


def check_exponent_approximation(n, Gamma, k, theta, case='kernel'):
    Deltas, gammas = get_gnm(Gamma, k, theta, n - 1, case)
    Gs = Deltas - 1j * gammas / 2
    aG, bG, cG, pG, qG = fit_exponential(Gs)
    ag, bg, cg, pg, qg = fit_exponential(gammas)
    import matplotlib.pyplot as plt
    ns = np.array(range(1, n))
    plt.scatter(ns, gammas)
    plt.scatter(ns, ag + bg * np.exp(pg * ns) + cg * np.exp(qg * ns))
    plt.title(r'$\gamma(r)$')
    plt.xlabel(r'$r$')
    plt.legend(['exact', 'exponent sum'])
    plt.show()
    plt.scatter(ns, np.real(Gs))
    plt.scatter(ns, np.real(bG * np.exp(pG * ns) + cG * np.exp(qG * ns)))
    plt.title(r'Re($G(r)$)')
    plt.xlabel(r'$r$')
    plt.legend(['exact', 'exponent sum'])
    plt.show()
    plt.scatter(ns, np.imag(Gs))
    plt.scatter(ns, np.imag(bG * np.exp(pG * ns) + cG * np.exp(qG * ns)))
    plt.legend(['exact', 'exponent sum'])
    plt.title(r'Im($G(r)$)')
    plt.xlabel(r'$r$')
    plt.show()


def get_photon_green_L_exp(n, Omega, Gamma, k, theta, sigma, case='kernel', with_a=False):
    d = 2
    if case == 'kernel_1d':
        mu = 3 / (2 * k * n)
        gamma_1d = mu * Gamma
        Ss = [get_single_L_term(Omega * np.exp(1j * k * i), gamma_1d, sigma) for i in range(n)]
        interacting_terms = [
                             [- gamma_1d / 2 * np.kron(I, sigma.T), np.exp(1j * k) * np.kron(I, I),
                              np.exp(1j * k) * np.kron(I, sigma)],
                             [- gamma_1d / 2 * np.kron(I, sigma), np.exp(1j * k) * np.kron(I, I),
                              np.exp(1j * k) * np.kron(I, sigma.T)],
                             # [- gamma_1d / 2 * np.kron(sigma.T, I), np.conj(np.exp(-1j * k)) * np.kron(I, I),
                             #  np.conj(np.exp(-1j * k)) * np.kron(sigma, I)],
                             # [- gamma_1d / 2 * np.kron(sigma, I), np.conj(np.exp(-1j * k)) * np.kron(I, I),
                             #  np.conj(np.exp(-1j * k)) * np.kron(sigma.T, I)],
                             [gamma_1d / 2 * np.kron(sigma, I), np.exp(1j * k) * np.kron(I, I), np.exp(1j * k) * np.kron(I, sigma)],
                             [gamma_1d / 2 * np.kron(sigma, I), np.exp(-1j * k) * np.kron(I, I), np.exp(-1j * k) * np.kron(I, sigma)],
                             [gamma_1d / 2 * np.kron(I, sigma), np.exp(1j * k) * np.kron(I, I), np.exp(1j * k) * np.kron(sigma, I)],
                             [gamma_1d / 2 * np.kron(I, sigma), np.exp(-1j * k) * np.kron(I, I), np.exp(-1j * k) * np.kron(sigma, I)],
                             ]
    elif case == 'kernel':
        S = get_single_L_term(Omega, Gamma, sigma)
        Ss = [np.copy(S) for i in range(n)]
        Deltas, gammas = get_gnm(Gamma, k, theta, n - 1, case)
        Gs = Deltas - 1j * gammas / 2
        aG, bG, cG, pG, qG = fit_exponential(Gs)
        ag, bg, cg, pg, qg = fit_exponential(gammas)
        interacting_terms = [[-1j * bG * np.kron(I, sigma.T), np.exp(pG) * np.kron(I, I), np.exp(pG) * np.kron(I, sigma)],
                             [-1j * bG * np.kron(I, sigma), np.exp(pG) * np.kron(I, I), np.exp(pG) * np.kron(I, sigma.T)],
                             [-1j * cG * np.kron(I, sigma.T), np.exp(qG) * np.kron(I, I), np.exp(qG) * np.kron(I, sigma)],
                             [-1j * cG * np.kron(I, sigma), np.exp(qG) * np.kron(I, I), np.exp(qG) * np.kron(I, sigma.T)],
                             [1j * np.conj(bG) * np.kron(sigma.T, I), np.exp(pG) * np.kron(I, I), np.exp(pG) * np.kron(sigma, I)],
                             [1j * np.conj(bG) * np.kron(sigma, I), np.exp(pG) * np.kron(I, I), np.exp(pG) * np.kron(sigma.T, I)],
                             [1j * np.conj(cG) * np.kron(sigma.T, I), np.exp(qG) * np.kron(I, I), np.exp(qG) * np.kron(sigma, I)],
                             [1j * np.conj(cG) * np.kron(sigma, I), np.exp(qG) * np.kron(I, I), np.exp(qG) * np.kron(sigma.T, I)],
                             [bg * np.kron(sigma, I), np.exp(pg) * np.kron(I, I), np.exp(pg) * np.kron(I, sigma)],
                             [bg * np.kron(I, sigma), np.exp(pg) * np.kron(I, I), np.exp(pg) * np.kron(sigma, I)],
                             [cg * np.kron(sigma, I), np.exp(qg) * np.kron(I, I), np.exp(qg) * np.kron(I, sigma)],
                             [cg * np.kron(I, sigma), np.exp(qg) * np.kron(I, I), np.exp(qg) * np.kron(sigma, I)],
                             [ag * np.kron(sigma, I), np.kron(I, I), np.kron(I, sigma)],
                             [ag * np.kron(I, sigma), np.kron(I, I), np.kron(sigma, I)]
                             ]
        if with_a:
            interacting_terms = interacting_terms + [
                [-1j * aG * np.kron(I, sigma), np.kron(I, I), np.kron(I, sigma.T)],
                [-1j * aG * np.kron(I, sigma.T), np.kron(I, I), np.kron(I, sigma)],
                [1j * np.conj(aG) * np.kron(sigma, I), np.kron(I, I), np.kron(sigma.T, I)],
                [1j * np.conj(aG) * np.kron(sigma.T, I), np.kron(I, I), np.kron(sigma, I)]
            ]
    operators_len = 2 + len(interacting_terms)
    nothing_yet_ind = operators_len - 1
    already_finished_ind = operators_len - 2
    left_tensor = np.zeros((d ** 2, d ** 2, 1, operators_len), dtype=complex)
    mid_tensor = np.zeros((d ** 2, d ** 2, operators_len, operators_len), dtype=complex)
    right_tensor = np.zeros((d ** 2, d ** 2, operators_len, 1), dtype=complex)
    left_tensor[:, :, 0, already_finished_ind] = Ss[0].T
    left_tensor[:, :, 0, nothing_yet_ind] = np.kron(I, I)
    mid_tensor[:, :, already_finished_ind, already_finished_ind] = np.kron(I, I)
    mid_tensor[:, :, nothing_yet_ind, nothing_yet_ind] = np.kron(I, I)
    right_tensor[:, :, nothing_yet_ind, 0] = Ss[n-1].T
    right_tensor[:, :, already_finished_ind, 0] = np.kron(I, I)
    for term_i in range(len(interacting_terms)):
        left_tensor[:, :, 0, term_i] = interacting_terms[term_i][0].T
        mid_tensor[:, :, nothing_yet_ind, term_i] = interacting_terms[term_i][0].T
        mid_tensor[:, :, term_i, term_i] = interacting_terms[term_i][1].T
        mid_tensor[:, :, term_i, already_finished_ind] = interacting_terms[term_i][2].T
        right_tensor[:, :, term_i, 0] = interacting_terms[term_i][2].T
    L = [tn.Node(left_tensor)]
    for i in range(1, n-1):
        curr_mid_tensor = np.copy(mid_tensor)
        curr_mid_tensor[:, :, nothing_yet_ind, already_finished_ind] = Ss[i].T
        L.append(tn.Node(curr_mid_tensor))
    L += [tn.Node(right_tensor)]
    return L


def get_photon_green_L(n, Omega, Gamma, k, theta, sigma, opt='NN', case='kernel', nearest_neighbors_num=1, exp_coeffs=[0]):
    d = 2
    S = get_single_L_term(Omega, Gamma, sigma)
    Deltas, gammas = get_gnm(Gamma, k, theta, nearest_neighbors_num, case)
    pairs = get_pair_L_terms(Deltas, gammas, nearest_neighbors_num, sigma)
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
    # for i in range(len(L) - 1):
    #     [r, l, te] = bops.svdTruncation(bops.contract(L[i], L[i+1], '3', '2'), [0, 1, 2], [3, 4, 5], '>>')
    #     L[i] = r
    #     L[i+1] = bops.permute(l, [1, 2, 0, 3])
    return L


def get_density_matrix_from_mps(psi, N):
    return bops.getExplicitVec(psi, d=4).reshape([2] * 2 * N).\
        transpose([2 * i for i in range(N)] + [2 * i + 1 for i in range(N)]).reshape([2**N, 2**N])


def filenames(newdir, case, N, Omega, nn_num, ti, bond_dim):
    state_filename = newdir + '/mid_state_' + case + '_N_' + str(N) \
        + '_Omega_' + str(Omega) + '_nn_' + str(nn_num) + '_ti_' + str(ti) + '_bond_' + str(bond_dim)
    data_filename = newdir + '/tdvp_N_' + str(N) \
                      + '_Omega_' + str(Omega) + '_nn_' + str(nn_num) + '_bond_' + str(bond_dim)
    return state_filename, data_filename


def get_sigma_z_expect(rho, N):
    res = 0
    for si in range(5, N-5):
        res += bops.getOverlap(rho,
                               [tn.Node(I) for i in range(si)] + [
                                   tn.Node(np.diag([1, -1]).reshape([1, d ** 2, 1]))]
                               + [tn.Node(I) for i in range(si + 1, N)])
    return res


def get_j_expect(rho, N, sigma):
    res = 0
    for si in range(N):
        res += bops.getOverlap(rho,
                    [tn.Node(I) for i in range(si)] + [tn.Node(np.matmul(sigma.T, sigma).reshape([1, d ** 2, 1]))]
                    + [tn.Node(I) for i in range(si + 1, N)])
        for sj in range(N):
            if si < sj:
                res += bops.getOverlap(rho,
                                    [tn.Node(I) for i in range(si)] + [tn.Node(sigma.T.reshape([1, d ** 2, 1]))]
                                    + [tn.Node(I) for i in range(si + 1, sj)] + [
                                        tn.Node(sigma.reshape([1, d ** 2, 1]))]
                                    + [tn.Node(I) for i in range(sj + 1, N)])
            elif sj < si:
                res += bops.getOverlap(rho,
                                    [tn.Node(I) for i in range(sj)] + [
                                        tn.Node(sigma.T.reshape([1, d ** 2, 1]))]
                                    + [tn.Node(I) for i in range(sj + 1, si)] + [
                                        tn.Node(sigma.reshape([1, d ** 2, 1]))]
                                    + [tn.Node(I) for i in range(si + 1, N)])
    return res


d = 2
Gamma = 1
sigma = np.array([[0, 0], [1, 0]])
I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Z = np.diag([1, -1])

N = int(sys.argv[1])
k = 2 * np.pi / 10
theta = 0
nn_num = int(sys.argv[2])
Omega = float(sys.argv[3]) / Gamma
case = sys.argv[4]
outdir = sys.argv[5]
timesteps = int(sys.argv[6])
dt = 1e-2
save_each = 10
results_to = sys.argv[7]
bond_dim = int(sys.argv[8])

newdir = outdir + '/' + case + '_N_' + str(N) + '_Omega_' + str(Omega) + '_nn_' + str(nn_num) + '_bd_' + str(bond_dim)
try:
    os.mkdir(newdir)
except FileExistsError:
    pass

if results_to == 'plot':
    import matplotlib.pyplot as plt

L_exp = get_photon_green_L_exp(N, Omega, Gamma, k, theta, sigma, case=case)
# L = get_photon_green_L(N, Omega, Gamma, k, theta, sigma, case=case, nearest_neighbors_num=nn_num)
psi = [tn.Node(np.array([1, 0, 0, 0]).reshape([1, d**2, 1])) for n in range(N)]
if N <= 6:
    print('starting exact tests')
    sigmas = [np.kron(np.eye(d**i), np.kron(sigma, np.eye(d**(N - i - 1)))) for i in range(N)]
    mu = 3 / (2 * k * N)
    gamma_1d = Gamma * mu
    mpo = bops.contract(L_exp[0], L_exp[1], '3', '2')
    gammas, deltas = get_gnm(Gamma, k, theta, nn_num, case)
    gammas = [Gamma] + list(gammas)
    deltas = [0] + list(deltas)
    for i in range(2, N):
        mpo = bops.contract(mpo, L_exp[i], [1 + 2 * i], '2')
    L_exp_mat = mpo.tensor.reshape([d] * 4 * N).transpose([0 + i * 4 for i in range(N)] +
                                                          [1 + i * 4 for i in range(N)] +
                                                          [2 + i * 4 for i in range(N)] +
                                                          [3 + i * 4 for i in range(N)]).reshape([d**(2 * N)] * 2).T
    H = np.zeros([2**N, 2**N], dtype=complex)
    for i in range(N):
        if case == 'kernel_1d':
            H += Omega * (np.exp(-1j * k * i) * sigmas[i] + np.exp(1j * k * i) * sigmas[i].conj().T)
        elif case == 'kernel':
            H += Omega * (sigmas[i] + sigmas[i].T)
        for j in range(N):
            if case == 'kernel_1d':
                H += (-1j) * gamma_1d / 2 * np.exp(1j * k * np.abs(i - j)) * np.matmul(sigmas[i].conj().T, sigmas[j])
            elif case == 'kernel':
                H += (deltas[np.abs(i - j)] - 1j * gammas[np.abs(i - j)] / 2) * np.matmul(sigmas[i].T, sigmas[j])
    L_mat = -1j * (np.kron(np.eye(2**N), H) - np.kron(H.conj(), np.eye(2**N)))
    c_plus = np.zeros((2**N, 2**N), dtype=complex)
    c_minus = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(N):
        c_plus += sigmas[i] * np.exp(1j * k * i)
        c_minus += sigmas[i] * np.exp(-1j * k * i)
        for j in range(N):
            if case == 'kernel_1d':
                L_mat += gamma_1d / 2 * np.exp(1j * k * np.abs(j - i)) * \
                     (np.kron(sigmas[i], sigmas[j]))
                L_mat += gamma_1d / 2 * np.exp(- 1j * k * np.abs(j - i)) * \
                         (np.kron(sigmas[i], sigmas[j]))
            elif case == 'kernel':
                L_mat += gammas[np.abs(i - j)] * (np.kron(sigmas[i], sigmas[j]))
    U = linalg.expm(dt * L_mat)
    where = np.where(np.round(L_mat - L_exp_mat, 14))
    diffs = [[bin(where[0][i]).split('b')[1].zfill(2 * N), bin(where[1][i]).split('b')[1].zfill(2 * N), np.round(L_exp_mat[where[0][i], where[1][i]], 14), np.round(L_mat[where[0][i], where[1][i]], 14)] for i in range(len(where[0]))]
    psi_exact = np.array([1] + [0] * (4**N - 1), dtype=complex)
    rho = psi_exact.reshape([2 ** N] * 2)
    J = np.zeros([2**N, 2**N], dtype=complex)
    for i in range(N):
        J += sigmas[i] * np.exp(-1j * k * i)
    JdJ = np.matmul(J.conj().T, J)
    Js = np.zeros(timesteps)
    for ti in range(timesteps):
        print(ti)
        # psi_exact = np.matmul(U, psi_exact)
        # rho = psi_exact.reshape([2**N] * 2)
        # rho += dt * (-1j * (np.matmul(H, rho) - np.matmul(rho, H.conj().T)) + \
        #        gamma_1d / 2 * np.matmul(c_minus, np.matmul(rho, c_minus.conj().T)))
        rho += dt * (-1j * (np.matmul(H, rho) - np.matmul(rho, H.conj().T)) + \
               gamma_1d / 2 * np.matmul(c_plus, np.matmul(rho, c_plus.conj().T)) + \
               gamma_1d / 2 * np.matmul(c_minus, np.matmul(rho, c_minus.conj().T)))
        print(np.round(rho.trace(), 14))
        Js[ti] = np.real(np.matmul(rho / rho.trace(), JdJ).trace())
    plt.plot(dt * np.array(range(timesteps)), Js)
    plt.show()

I = np.eye(2).reshape([1, d**2, 1])
J_expect = np.zeros(timesteps, dtype=complex)
bond_dims = np.zeros(timesteps, dtype=complex)

# projectors_left, projectors_right = tdvp.get_initial_projectors(psi, L)
hl_2_exp, hr_2_exp = tdvp.get_initial_projectors(psi, L_exp)

psi_2_exp = bops.copyState(psi)
psi_1_exp = bops.copyState(psi)
hl_1_exp = bops.copyState(hl_2_exp)
hr_1_exp = bops.copyState(hr_2_exp)

runtimes_2_exp = np.zeros(timesteps)
runtimes_1_exp = np.zeros(timesteps)

tes_2_exp = np.zeros(timesteps)
tes_1_exp = np.zeros(timesteps)

JdJ_2_exp = np.zeros(timesteps)
JdJ_1_exp = np.zeros(timesteps)

sigmaz_2_exp = np.zeros(timesteps)
sigmaz_1_exp = np.zeros(timesteps)

initial_ti = 0
for file in os.listdir(newdir):
    if file[-7:] == '_1s_exp':
        ti = int(file.split('_')[file.split('_').index('ti') + 1])
        if ti + 1 > initial_ti:
            initial_ti = ti + 1
            data = pickle.load(open(newdir + '/' + file, 'rb'))
            [ti, psi, projectors_left, projectors_right] = data[:4]
            runtimes_1_exp[:len(data[4])] = data[4]
            if len(data) > 5: tes_1_exp[:len(data[5])] = data[5]
if initial_ti > 0:
    state_filename, data_filename = filenames(newdir, case, N, Omega, nn_num, initial_ti - 1, bond_dim)
    data = pickle.load(open(state_filename + '_2s_exp', 'rb'))
    [_, psi_2_exp, hl_2_exp, hr_2_exp] = data[:4]
    runtimes_2_exp[:len(data[4])] = data[4]
    if len(data) > 5: tes_2_exp[:len(data[5])] = data[5]

for ti in range(initial_ti, timesteps):
    print('---')
    print(ti)
    if ti > 0 and ti % save_each != 1:
        tstart = time.time()
        psi_2_exp = bops.copyState(psi_1_exp)
        hl_2_exp = bops.copyState(hl_1_exp)
        hr_2_exp = bops.copyState(hr_1_exp)
        tes_2_exp[ti] = tdvp.tdvp_sweep(psi_2_exp, L_exp, hl_2_exp, hr_2_exp, dt / 2, max_bond_dim=bond_dim, num_of_sites=2)
        tf = time.time()
        runtimes_2_exp[ti] = tf - tstart

        old_state_filename, old_data_filename = filenames(newdir, case, N, Omega, nn_num, ti - 1, bond_dim)
        os.remove(old_state_filename + '_2s_exp')
        os.remove(old_state_filename + '_1s_exp')
    tstart = time.time()
    tes_1_exp[ti] = tdvp.tdvp_sweep(psi_1_exp, L_exp, hl_1_exp, hr_1_exp, dt / 2, max_bond_dim=bond_dim, num_of_sites=1)
    JdJ_1_exp[ti] = get_j_expect(psi_1_exp, N, sigma)
    sigmaz_1_exp[ti] = get_sigma_z_expect(psi_1_exp, N)
    tf = time.time()
    runtimes_1_exp[ti] = tf - tstart
    print('times = ' + str([runtimes_2_exp[ti], runtimes_1_exp[ti]]))
    state_filename, data_filename = filenames(newdir, case, N, Omega, nn_num, ti, bond_dim)
    with open(state_filename + '_2s_exp', 'wb') as f:
        pickle.dump([ti, psi_2_exp, hl_2_exp, hr_2_exp, runtimes_2_exp], f)
    with open(state_filename + '_1s_exp', 'wb') as f:
        pickle.dump([ti, psi_1_exp, hl_1_exp, hr_1_exp, runtimes_1_exp, tes_1_exp, JdJ_1_exp, sigmaz_1_exp], f)

if results_to == 'plot':
    plt.plot(dt * np.array(range(timesteps)), JdJ_1_exp)
    plt.show()
    print('plot')
    J_expect_1 = np.zeros(int(timesteps / save_each))
    J_expect_2 = np.zeros(int(timesteps / save_each))
    J_expect_1_corrected = np.zeros(int(timesteps / save_each))
    J_expect_1_exp = np.zeros(int(timesteps / save_each))
    J_expect_2_exp = np.zeros(int(timesteps / save_each))
    runtimes_1 = np.zeros(int(timesteps / save_each))
    runtimes_2 = np.zeros(int(timesteps / save_each))
    runtimes_1_corrected = np.zeros(int(timesteps / save_each))
    bd_1 = np.zeros(int(timesteps / save_each))
    bd_2 = np.zeros(int(timesteps / save_each))
    bd_1_corrected = np.zeros(int(timesteps / save_each))
    for ti in range(0, int(timesteps / save_each)):
        print(ti)
        state_filename, data_filename = filenames(newdir, case, N, Omega, nn_num, ti * save_each, bond_dim)
        # psi = pickle.load(open(state_filename + '_1s', 'rb'))[1]
        # bd_1[ti] = psi[int(N/2)][0].dimension
        # runtimes_1 = pickle.load(open(state_filename + '_1s', 'rb'))[-1]
        # J_expect_1[ti] = get_j_expect(psi, N, sigma)
        # psi_2 = pickle.load(open(state_filename + '_2s', 'rb'))[1]
        # bd_2[ti] = psi_2[int(N/2)][0].dimension
        # runtimes_2 = pickle.load(open(state_filename + '_2s', 'rb'))[-1]
        # J_expect_2[ti] = get_j_expect(psi_2, N, sigma)
        # psi_1_corrected_w = pickle.load(open(state_filename + '_1s_low_preselection', 'rb'))[1]
        # bd_1_corrected[ti] = psi_1_corrected_w[int(N/2)][0].dimension
        # runtimes_1_corrected = pickle.load(open(state_filename + '_1s_low_preselection', 'rb'))[-1]
        # J_expect_1_corrected[ti] = get_j_expect(psi_1_corrected_w, N, sigma)
        psi_1_exp = pickle.load(open(state_filename + '_1s_exp', 'rb'))[1]
        J_expect_1_exp[ti] = get_j_expect(psi_1_exp, N, sigma)
        psi_2_exp = pickle.load(open(state_filename + '_2s_exp', 'rb'))[1]
        J_expect_2_exp[ti] = get_j_expect(psi_2_exp, N, sigma)
    # plt.plot(np.array(range(int(timesteps / save_each))) * dt * save_each, J_expect_1)
    # plt.plot(np.array(range(int(timesteps / save_each))) * dt * save_each, J_expect_2, '--')
    # plt.plot(np.array(range(int(timesteps / save_each))) * dt * save_each, J_expect_1_corrected, ':')
    plt.plot(np.array(range(int(timesteps / save_each))) * dt * save_each, J_expect_1_exp, ':')
    plt.plot(np.array(range(int(timesteps / save_each))) * dt * save_each, J_expect_2_exp, '--')
    plt.show()
    # plt.plot(runtimes_1)
    # plt.plot(runtimes_2)
    # plt.plot(runtimes_1_corrected)
    # plt.show()
    # plt.plot(bd_1)
    # plt.plot(bd_2)
    # plt.plot(bd_1_corrected)
    # plt.show()