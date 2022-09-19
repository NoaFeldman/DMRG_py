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
    return -1j * Omega * (np.kron(np.eye(d), sigma + sigma.T) - np.kron(sigma + sigma.T, np.eye(d))) \
        + Gamma * (np.kron(sigma, sigma)
                   - 0.5 * (np.kron(np.matmul(sigma.T, sigma), np.eye(d)) + np.kron(np.eye(d), np.matmul(sigma.T, sigma))))


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
    S = get_single_L_term(Omega, Gamma, sigma)
    Deltas, gammas = get_gnm(Gamma, k, theta, n - 1, case)
    Gs = Deltas - 1j * gammas / 2
    aG, bG, cG, pG, qG = fit_exponential(Gs)
    ag, bg, cg, pg, qg = fit_exponential(gammas)
    interacting_terms = [[-1j * bG * np.kron(I, sigma.T), np.exp(pG) * np.kron(I, I), np.exp(pG) * np.kron(I, sigma)],
                         [-1j * bG * np.kron(I, sigma), np.exp(pG) * np.kron(I, I), np.exp(pG) * np.kron(I, sigma.T)],
                         [-1j * cG * np.kron(I, sigma.T), np.exp(qG) * np.kron(I, I), np.exp(qG) * np.kron(I, sigma)],
                         [-1j * cG * np.kron(I, sigma), np.exp(qG) * np.kron(I, I), np.exp(qG) * np.kron(I, sigma.T)],
                         [1j * np.conj(bG) * np.kron(sigma.T, I), np.conj(np.exp(pG)) * np.kron(I, I), np.conj(np.exp(pG)) * np.kron(sigma, I)],
                         [1j * np.conj(bG) * np.kron(sigma, I), np.conj(np.exp(pG)) * np.kron(I, I), np.conj(np.exp(pG)) * np.kron(sigma.T, I)],
                         [1j * np.conj(cG) * np.kron(sigma.T, I), np.conj(np.exp(qG)) * np.kron(I, I), np.conj(np.exp(qG)) * np.kron(sigma, I)],
                         [1j * np.conj(cG) * np.kron(sigma, I), np.conj(np.exp(qG)) * np.kron(I, I), np.conj(np.exp(qG)) * np.kron(sigma.T, I)],
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
    left_tensor[:, :, 0, already_finished_ind] = S.T
    left_tensor[:, :, 0, nothing_yet_ind] = np.kron(I, I)
    mid_tensor[:, :, nothing_yet_ind, already_finished_ind] = S.T
    mid_tensor[:, :, already_finished_ind, already_finished_ind] = np.kron(I, I)
    mid_tensor[:, :, nothing_yet_ind, nothing_yet_ind] = np.kron(I, I)
    right_tensor[:, :, nothing_yet_ind, 0] = S.T
    right_tensor[:, :, already_finished_ind, 0] = np.kron(I, I)
    for term_i in range(len(interacting_terms)):
        left_tensor[:, :, 0, term_i] = interacting_terms[term_i][0].T
        mid_tensor[:, :, nothing_yet_ind, term_i] = interacting_terms[term_i][0].T
        mid_tensor[:, :, term_i, term_i] = interacting_terms[term_i][1].T
        mid_tensor[:, :, term_i, already_finished_ind] = interacting_terms[term_i][2].T
        right_tensor[:, :, term_i, 0] = interacting_terms[term_i][2].T
    return [tn.Node(left_tensor)] + [tn.Node(mid_tensor) for si in range(n - 2)] + [tn.Node(right_tensor)]


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

L_exp = get_photon_green_L_exp(N, Omega, Gamma, k, theta, sigma)
L = get_photon_green_L(N, Omega, Gamma, k, theta, sigma, case=case, nearest_neighbors_num=nn_num)
psi = [tn.Node(np.array([1, 0, 0, 0]).reshape([1, d**2, 1])) for n in range(N)]
if N <= 12:
    print(nn_num)
    rhos = []
    Deltas, gammas = get_gnm(Gamma, k, theta, nn_num, case)
    if case == 'kernel':
        Deltas = np.array([0] + list(Deltas))
        gammas = np.array([Gamma] + list(gammas))
    H_eff_exact = np.zeros((d**(N), d**(N)), dtype=complex)
    sigmas = []
    for i in range(N):
        sigmas.append(np.kron(np.eye(d**(i)), np.kron(sigma, np.eye(d**(N - i - 1)))))

    for n in range(N):
        H_eff_exact += Omega * (sigmas[n] + sigmas[n].T)
        for m in range(N):
            if np.abs(m - n) <= nn_num:
                H_eff_exact += (Deltas[np.abs(n - m)] - 1j * gammas[np.abs(n - m)] / 2) * \
                               np.matmul(sigmas[n], sigmas[m].T)
                # L_exact += (-1j * Deltas[np.abs(m - n)] - 0.5 * gammas[np.abs(m - n)]) * \
                #            np.kron(np.eye(d**N), np.matmul(sigmas[n].T, sigmas[m]))
                # L_exact += (1j * Deltas[np.abs(m - n)] - 0.5 * gammas[np.abs(m - n)]) * \
                #            np.kron(np.matmul(sigmas[n].T, sigmas[m]), np.eye(d**N))
                # L_exact += gammas[np.abs(m - n)] * np.kron(sigmas[n], sigmas[m])

    # explicit_L = bops.contract(L[0], L[1], '3', '2')
    # # explicit_L_exp = bops.contract(L_exp[0], L_exp[1], '3', '2')
    # explicit_rho = bops.contract(psi[0], psi[1], '2', '0')
    # for ni in range(2, N):
    #     explicit_L = bops.contract(explicit_L, L[ni], [2 * ni + 1], '2')
    #     # explicit_L_exp = bops.contract(explicit_L_exp, L_exp[ni], [2 * ni + 1], '2')
    #     explicit_rho = bops.contract(explicit_rho, psi[ni], [ni + 1], '0')
    # L_mat = explicit_L.tensor.reshape([d] * 4 * N).transpose([4 * i for i in range(N)]
    #                                                          + [1 + 4 * i for i in range(N)]
    #                                                          + [2 + 4 * i for i in range(N)]
    #                                                          + [3 + 4 * i for i in range(N)])\
    #     .reshape([d**(2 * N), d**(2 * N)]).T
    # L_exp_mat = explicit_L_exp.tensor.reshape([d] * 4 * N).transpose([4 * i for i in range(N)]
    #                                                          + [1 + 4 * i for i in range(N)]
    #                                                          + [2 + 4 * i for i in range(N)]
    #                                                          + [3 + 4 * i for i in range(N)])\
    #     .reshape([d**(2 * N), d**(2 * N)]).T
    Deltas, gammas = get_gnm(Gamma, k, theta, N - 1, case)
    Gs = Deltas - 1j * gammas / 2
    aG, bG, cG, pG, qG = fit_exponential(Gs)
    ag, bg, cg, pg, qg = fit_exponential(gammas)
    # L_exp_exact = np.zeros((d**(2 * N), d**(2 * N)), dtype=complex)
    # for n in range(N):
    #     L_exp_exact += - 0.5 * np.kron(np.eye(d ** N), np.matmul(sigmas[n].T, sigmas[n]))
    #     L_exp_exact += - 0.5 * np.kron(np.matmul(sigmas[n].T, sigmas[n]), np.eye(d ** N))
    #     L_exp_exact += np.kron(sigmas[n], sigmas[n])
    #     for m in range(N):
    #         if m != n and np.abs(m - n) <= nn_num:
    #             L_exp_exact += (-1j * (aG + bG * np.exp(pG * np.abs(m - n)) + cG * np.exp(qG * np.abs(m - n)))) * \
    #                        np.kron(np.eye(d ** N), np.matmul(sigmas[n].T, sigmas[m]))
    #             L_exp_exact += (1j * np.conj(aG + bG * np.exp(pG * np.abs(m - n)) + cG * np.exp(qG * np.abs(m - n)))) * \
    #                    np.kron(np.matmul(sigmas[n].T, sigmas[m]), np.eye(d ** N))
    #             L_exp_exact += (ag + bg * np.exp(pg * np.abs(m - n)) + cg * np.exp(qg * np.abs(m - n))) * np.kron(sigmas[n], sigmas[m])
    rho_mat = np.diag([1] + [0] * (d**N - 1))
    rho_vec = bops.getExplicitVec(psi, d**2)
    rho_vec_exp = np.copy(rho_vec)
    rho_vec_exp_exact = np.copy(rho_vec)
    # evolver_L = linalg.expm(L_exact * dt)
    J_expect_L = np.zeros(timesteps)
    # evolver_L_exp = linalg.expm(L_exp_mat * dt)
    J_expect_L_exp = np.zeros(timesteps)
    # evolver_L_exp_exact = linalg.expm(L_exp_exact * dt)
    J_expect_L_exp_exact = np.zeros(timesteps)

    z_inds = [[i + d**N * i,
               2 * bin(i).split('b')[1].count('1') - N]
              for i in range(d**N)]
    J = np.zeros(sigmas[0].shape)
    for i in range(N):
        J += sigmas[i]
    JdJ = np.matmul(J.conj().T, J)
    for ti in range(timesteps):
        print(ti)
        J_expect_L[ti] = np.abs(np.trace(np.matmul(JdJ, rho_mat)))
        addition = 1j * (np.matmul(H_eff_exact, rho_mat) - np.matmul(rho_mat, H_eff_exact.conj().T))
        for n in range(N):
            for m in range(N):
                if np.abs(m - n) <= nn_num:
                    addition += gammas[np.abs(n - m)] * np.matmul(sigmas[n], np.matmul(rho_mat, sigmas[m].T))
        rho_mat += dt * addition
        # J_expect_L_exp[ti] = np.abs(np.trace(np.matmul(JdJ, rho_vec_exp.reshape([d**N, d**N]))))
        # J_expect_L_exp_exact[ti] = np.abs(np.trace(np.matmul(JdJ, rho_vec_exp_exact.reshape([d**N, d**N]))))
        # rho_vec = np.matmul(evolver_L, rho_vec)
        # rho_vec_exp = np.matmul(evolver_L_exp, rho_vec_exp)
        # rho_vec_exp_exact = np.matmul(evolver_L_exp_exact, rho_vec_exp_exact)
        rhos.append(rho_vec)

    if results_to == 'plot':
        plt.plot(np.array(range(int(timesteps))) * dt, J_expect_L)
        plt.plot(np.array(range(int(timesteps))) * dt, J_expect_L_exp)
        plt.plot(np.array(range(int(timesteps))) * dt, J_expect_L_exp_exact)
        plt.show()
    else:
        with open(outdir + '/explicit_J_expect', 'wb') as f:
            pickle.dump(J_expect_L, f)

I = np.eye(2).reshape([1, d**2, 1])
J_expect = np.zeros(timesteps, dtype=complex)
bond_dims = np.zeros(timesteps, dtype=complex)

projectors_left, projectors_right = tdvp.get_initial_projectors(psi, L)
hl_2_exp, hr_2_exp = tdvp.get_initial_projectors(psi, L_exp)

psi_1_corrected_w = bops.copyState(psi)
hl_1_corrected = bops.copyState(projectors_left)
hr_1_corrected = bops.copyState(projectors_right)
psi_2 = bops.copyState(psi)
hl_2 = bops.copyState(projectors_left)
hr_2 = bops.copyState(projectors_right)
psi_2_exp = bops.copyState(psi)
psi_1_exp = bops.copyState(psi)
hl_1_exp = bops.copyState(hl_2_exp)
hr_1_exp = bops.copyState(hr_2_exp)

runtimes_1 = np.zeros(timesteps)
runtimes_1_corrected = np.zeros(timesteps)
runtimes_2 = np.zeros(timesteps)
runtimes_2_exp = np.zeros(timesteps)
runtimes_1_exp = np.zeros(timesteps)

tes_1 = np.zeros(timesteps)
tes_1_corrected = np.zeros(timesteps)
tes_2 = np.zeros(timesteps)
tes_2_exp = np.zeros(timesteps)
tes_1_exp = np.zeros(timesteps)

initial_ti = 0
for file in os.listdir(newdir):
    if file[-3:] == '_1s':
        ti = int(file.split('_')[file.split('_').index('ti') + 1])
        if ti + 1 > initial_ti:
            initial_ti = ti + 1
            data = pickle.load(open(newdir + '/' + file, 'rb'))
            [ti, psi, projectors_left, projectors_right] = data[:4]
            runtimes_1[:len(data[4])] = data[4]
            if len(data) > 5: tes_1[:len(data[5])] = data[5]
if initial_ti > 0:
    state_filename, data_filename = filenames(newdir, case, N, Omega, nn_num, initial_ti - 1, bond_dim)
    data = pickle.load(open(state_filename + '_2s', 'rb'))
    [_, psi_2, hl_2, hr_2] = data[:4]
    runtimes_2[:len(data[4])] = data[4]
    if len(data) > 5: tes_2[:len(data[5])] = data[5]
    data = pickle.load(open(state_filename + '_1s_low_preselection', 'rb'))
    [_, psi_1_corrected_w, hl_1_corrected, hr_1_corrected] = data[:4]
    runtimes_1_corrected[:len(data[4])] = data[4]
    if len(data) > 5: tes_1_corrected[:len(data[5])] = data[5]
    data = pickle.load(open(state_filename + '_2s_exp', 'rb'))
    [_, psi_2_exp, hl_2_exp, hr_2_exp] = data[:4]
    runtimes_2_exp[:len(data[4])] = data[4]
    if len(data) > 5: tes_2_exp[:len(data[5])] = data[5]
    data = pickle.load(open(state_filename + '_1s_exp', 'rb'))
    [_, psi_1_exp, hl_1_exp, hr_1_exp] = data[:4]
    runtimes_1_exp[:len(data[4])] = data[4]
    if len(data) > 5: tes_1_exp[:len(data[5])] = data[5]

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
        os.remove(old_state_filename + '_1s')
        os.remove(old_state_filename + '_2s')
        os.remove(old_state_filename + '_1s_low_preselection')
        os.remove(old_state_filename + '_2s_exp')
        os.remove(old_state_filename + '_1s_exp')
    # tstart = time.time()
    # tes_1[ti] = tdvp.tdvp_sweep(psi, L, projectors_left, projectors_right, dt / 2, max_bond_dim=bond_dim, num_of_sites=1)
    # tf = time.time()
    # runtimes_1[ti] = tf - tstart
    # tstart = time.time()
    # tes_1_corrected[ti] = tdvp.tdvp_sweep(psi_1_corrected_w, L, hl_1_corrected, hr_1_corrected, dt / 2,
    #                                       max_bond_dim=bond_dim, num_of_sites=1, max_trunc=12)
    # tf = time.time()
    # runtimes_1_corrected[ti] = tf - tstart
    # tstart = time.time()
    # tes_2[ti] = tdvp.tdvp_sweep(psi_2, L, hl_2, hr_2, dt / 2, max_bond_dim=bond_dim, num_of_sites=2)
    # tf = time.time()
    # runtimes_2[ti] = tf - tstart
    # tstart = time.time()
    # tes_2_exp[ti] = tdvp.tdvp_sweep(psi_2_exp, L_exp, hl_2_exp, hr_2_exp, dt / 2, max_bond_dim=bond_dim, num_of_sites=2)
    # tf = time.time()
    # runtimes_2_exp[ti] = tf - tstart
    tstart = time.time()
    tes_1_exp[ti] = tdvp.tdvp_sweep(psi_1_exp, L_exp, hl_1_exp, hr_1_exp, dt / 2, max_bond_dim=bond_dim, num_of_sites=1)
    tf = time.time()
    runtimes_1_exp[ti] = tf - tstart
    print('times = ' + str([runtimes_1[ti], runtimes_2[ti], runtimes_1_corrected[ti], runtimes_2_exp[ti], runtimes_1_exp[ti]]))
    state_filename, data_filename = filenames(newdir, case, N, Omega, nn_num, ti, bond_dim)
    with open(state_filename + '_1s', 'wb') as f:
        pickle.dump([ti, psi, projectors_left, projectors_right, runtimes_1, tes_1], f)
    with open(state_filename + '_1s_low_preselection', 'wb') as f:
        pickle.dump([ti, psi_1_corrected_w, hl_1_corrected, hr_1_corrected, runtimes_1_corrected, tes_1_corrected], f)
    with open(state_filename + '_2s', 'wb') as f:
        pickle.dump([ti, psi_2, hl_2, hr_2, runtimes_2, tes_2], f)
    with open(state_filename + '_2s_exp', 'wb') as f:
        pickle.dump([ti, psi_2_exp, hl_2_exp, hr_2_exp, runtimes_2_exp], f)
    with open(state_filename + '_1s_exp', 'wb') as f:
        pickle.dump([ti, psi_1_exp, hl_1_exp, hr_1_exp, runtimes_1_exp, tes_1_exp], f)

if results_to == 'plot':
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
        state_filename, data_filename = filenames(newdir, case, N, Omega, nn_num, ti * save_each, bond_dim)
        psi = pickle.load(open(state_filename + '_1s', 'rb'))[1]
        bd_1[ti] = psi[int(N/2)][0].dimension
        runtimes_1 = pickle.load(open(state_filename + '_1s', 'rb'))[-1]
        J_expect_1[ti] = get_j_expect(psi, N, sigma)
        psi_2 = pickle.load(open(state_filename + '_2s', 'rb'))[1]
        bd_2[ti] = psi_2[int(N/2)][0].dimension
        runtimes_2 = pickle.load(open(state_filename + '_2s', 'rb'))[-1]
        J_expect_2[ti] = get_j_expect(psi_2, N, sigma)
        psi_1_corrected_w = pickle.load(open(state_filename + '_1s_low_preselection', 'rb'))[1]
        bd_1_corrected[ti] = psi_1_corrected_w[int(N/2)][0].dimension
        runtimes_1_corrected = pickle.load(open(state_filename + '_1s_low_preselection', 'rb'))[-1]
        J_expect_1_corrected[ti] = get_j_expect(psi_1_corrected_w, N, sigma)
        psi_1_exp = pickle.load(open(state_filename + '_1s_exp', 'rb'))[1]
        J_expect_1_exp[ti] = get_j_expect(psi_1_exp, N, sigma)
        psi_2_exp = pickle.load(open(state_filename + '_2s_exp', 'rb'))[1]
        J_expect_2_exp[ti] = get_j_expect(psi_2_exp, N, sigma)
    plt.plot(np.array(range(int(timesteps / save_each))) * dt * save_each, J_expect_1)
    plt.plot(np.array(range(int(timesteps / save_each))) * dt * save_each, J_expect_2, '--')
    # plt.plot(np.array(range(int(timesteps / save_each))) * dt * save_each, J_expect_1_corrected, ':')
    # plt.plot(np.array(range(int(timesteps / save_each))) * dt * save_each, J_expect_1_exp, ':')
    # plt.plot(np.array(range(int(timesteps / save_each))) * dt * save_each, J_expect_2_exp, ':')
    plt.show()
    # plt.plot(runtimes_1)
    # plt.plot(runtimes_2)
    # plt.plot(runtimes_1_corrected)
    # plt.show()
    # plt.plot(bd_1)
    # plt.plot(bd_2)
    # plt.plot(bd_1_corrected)
    # plt.show()