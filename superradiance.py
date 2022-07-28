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


def get_photon_green_L_exp(n, Omega, Gamma, k, theta, sigma, case='kernel', nearest_neighbors_num=1, exp_coeffs=[0]):
    d = 2
    S = get_single_L_term(Omega, Gamma, sigma)
    Deltas, gammas = get_gnm(Gamma, k, theta, nearest_neighbors_num, case)
    ad, bd, cd, pd, qd = fit_exponential(Deltas)
    print('neglecting constant in Deltas fit:' + str(ad / np.max(Deltas)))
    ag, bg, cg, pg, qg = fit_exponential(Deltas)
    print('neglecting constant in Gammas fit:' + str(ag / np.max(Deltas)))
    interacting_terms = [[-1j * bd * np.kron(I, sigma.T)], [np.exp(pd) * I], [np.exp(pd) * np.kron(I, sigma)] +
                         [-1j * cd * np.kron(I, sigma.T)], [np.exp(qd) * I], [np.exp(qd) * np.kron(I, sigma)] +
                         [-1j * bd * np.kron(I, sigma)], [np.exp(pd) * I], [np.exp(pd) * np.kron(I, sigma.T)] +
                         [-1j * cd * np.kron(I, sigma)], [np.exp(qd) * I], [np.exp(qd) * np.kron(I, sigma.T)] +
                         [-0.5 * bg * np.kron(I, sigma.T)], [np.exp(pg) * I], [np.exp(pg) * np.kron(I, sigma)] +
                         [-0.5 * cg * np.kron(I, sigma.T)], [np.exp(qg) * I], [np.exp(qg) * np.kron(I, sigma)] +
                         [-0.5 * bg * np.kron(I, sigma)], [np.exp(pg) * I], [np.exp(pg) * np.kron(I, sigma.T)] +
                         [-0.5 * cg * np.kron(I, sigma)], [np.exp(qg) * I], [np.exp(qg) * np.kron(I, sigma.T)] +
                         [1j * bd * np.kron(sigma.T, I)], [np.exp(pd) * I], [np.exp(pd) * np.kron(sigma, I)] +
                         [1j * cd * np.kron(sigma.T, I)], [np.exp(qd) * I], [np.exp(qd) * np.kron(sigma, I)] +
                         [1j * bd * np.kron(sigma, I)], [np.exp(pd) * I], [np.exp(pd) * np.kron(sigma.T, I)] +
                         [1j * cd * np.kron(sigma, I)], [np.exp(qd) * I], [np.exp(qd) * np.kron(sigma.T, I)] +
                         [0.5 * bg * np.kron(sigma.T, I)], [np.exp(pg) * I], [np.exp(pg) * np.kron(sigma, I)] +
                         [0.5 * cg * np.kron(sigma.T, I)], [np.exp(qg) * I], [np.exp(qg) * np.kron(sigma, I)] +
                         [0.5 * bg * np.kron(sigma, I)], [np.exp(pg) * I], [np.exp(pg) * np.kron(sigma.T, I)] +
                         [0.5 * cg * np.kron(sigma, I)], [np.exp(qg) * I], [np.exp(qg) * np.kron(sigma.T, I)] +
                         [bg * np.kron(sigma, I)], [np.exp(pg) * I], [np.exp(pg) * np.kron(I, sigma)] +
                         [cg * np.kron(sigma, I)], [np.exp(qg) * I], [np.exp(qg) * np.kron(I, sigma)] +
                         [bg * np.kron(I, sigma)], [np.exp(pg) * I], [np.exp(pg) * np.kron(sigma, I)] +
                         [cg * np.kron(I, sigma)], [np.exp(qg) * I], [np.exp(qg) * np.kron(sigma, I)]
                         ]
    operators_len = 2 + 20 # TODO I need more here - also for the mid-Is in the interaction term
    nothing_yet_ind = operators_len - 1
    left_tensor = np.zeros((d ** 2, d ** 2, 1, operators_len), dtype=complex)
    mid_tensor = np.zeros((d ** 2, d ** 2, operators_len, operators_len), dtype=complex)
    right_tensor = np.zeros((d ** 2, d ** 2, operators_len, 1), dtype=complex)
    left_tensor[:, :, 0, 0] = S.T
    left_tensor[:, :, 0, nothing_yet_ind] = I
    mid_tensor[:, :, nothing_yet_ind, 0] = S.T
    mid_tensor[:, :, 0, 0] = I
    mid_tensor[:, :, nothing_yet_ind, nothing_yet_ind] = I
    right_tensor[:, :, nothing_yet_ind, 0] = S.T
    right_tensor[:, :, 0, 0] = I
    for term_i in range(len(interacting_terms)):
        left_tensor[:, :, 0, term_i + 1] = interacting_terms[term_i][0]
        left_tensor[:, :, term_i + 1, ] = interacting_terms[term_i][0]


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
    for i in range(len(L) - 1):
        [r, l, te] = bops.svdTruncation(bops.contract(L[i], L[i+1], '3', '2'), [0, 1, 2], [3, 4, 5], '>>')
        L[i] = r
        L[i+1] = bops.permute(l, [1, 2, 0, 3])
    return L


def filenames(newdir, case, N, Omega, nn_num, ti, method, bond_dim):
    if method == 'tdvp':
        state_filename = newdir + '/mid_state_' + case + '_N_' + str(N) \
            + '_Omega_' + str(Omega) + '_nn_' + str(nn_num) + '_ti_' + str(ti) + '_bond_' + str(bond_dim)
        data_filename = newdir + '/tdvp_' + case + '_N_' + str(N) \
                          + '_Omega_' + str(Omega) + '_nn_' + str(nn_num) + '_bond_' + str(bond_dim)
    elif method == 'swap':
        state_filename = newdir + '/swap_mid_state_' + case + '_N_' + str(N) \
                         + '_Omega_' + str(Omega) + '_nn_' + str(nn_num) + '_ti_' + str(ti) + '_bond_' + str(bond_dim)
        data_filename = newdir + '/swap_' + case + '_N_' + str(N) \
                        + '_Omega_' + str(Omega) + '_nn_' + str(nn_num) + '_bond_' + str(bond_dim)
    return state_filename, data_filename

d = 2
Gamma = 1
sigma = np.array([[0, 0], [1, 0]])
I = np.eye(2).reshape([1, d ** 2, 1])
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
T = 10
dt = T / timesteps
save_each = 10
results_to = sys.argv[7]
sim_method = sys.argv[8]
bond_dim = int(sys.argv[9])

newdir = outdir + '/' + sim_method + '_' + case + '_N_' + str(N) + '_Omega_' + str(Omega) + '_nn_' + str(nn_num) + '_timesteps_' + str(timesteps)
try:
    os.mkdir(newdir)
except FileExistsError:
    pass

if results_to == 'plot':
    import matplotlib.pyplot as plt

L = get_photon_green_L(N, Omega, Gamma, k, theta, sigma, case=case, nearest_neighbors_num=nn_num)
psi = [tn.Node(np.array([1, 0, 0, 0]).reshape([1, d**2, 1])) for n in range(N)]
if N <= 6:
    Deltas, gammas = get_gnm(Gamma, k, theta, nn_num, case)
    if case == 'kernel':
        Deltas = np.array([0] + list(Deltas))
        gammas = np.array([Gamma] + list(gammas))
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
        .reshape([d**(2 * N), d**(2 * N)]).T
    rho_vec = bops.getExplicitVec(psi, d**2)
    evolver_L = linalg.expm(L_mat * dt)
    J_expect_L = np.zeros(timesteps)
    z_inds = [[i + d**N * i,
               2 * bin(i).split('b')[1].count('1') - N]
              for i in range(d**N)]
    J = np.zeros(sigmas[0].shape)
    for i in range(N):
        J += sigmas[i]
    JdJ = np.matmul(J.conj().T, J)
    for ti in range(timesteps):
        J_expect_L[ti] = np.abs(np.trace(np.matmul(JdJ, rho_vec.reshape([d**N, d**N]))))
        rho_vec = np.matmul(evolver_L, rho_vec)

    if results_to == 'plot':
        plt.plot(J_expect_L)
    else:
        with open(outdir + '/explicit_J_expect', 'wb') as f:
            pickle.dump(J_expect_L, f)

I = np.eye(2).reshape([1, d**2, 1])
J_expect = np.zeros(timesteps, dtype=complex)
bond_dims = np.zeros(timesteps, dtype=complex)


projectors_left, projectors_right = tdvp.get_initial_projectors(psi, L)
if sim_method == 'swap':
    swap_op = tn.Node(np.eye(d**4).reshape([d**2] * 4).transpose([0, 1, 3, 2]))
    trotter_single_op = swap.get_single_trotter_op(get_single_L_term(Omega, Gamma, sigma).T, 1j * dt)
    pairs = get_pair_L_terms(Deltas, gammas, nn_num, sigma)
    terms = []
    for ni in range(nn_num):
        curr = np.kron(pairs[0][0][ni], pairs[0][1])
        for pi in range(1, len(pairs)):
            curr += np.kron(pairs[pi][0][ni], pairs[pi][1])
        terms.append(curr)
    neighbor_trotter_ops = swap.get_neighbor_trotter_ops([term.T for term in terms], 1j * dt, d**2)

sigma_expect = np.zeros(timesteps)
sigma_T_expect = np.zeros(timesteps)
sigma_X_expect = np.zeros(timesteps)
sigma_Z_expect = np.zeros(timesteps)
J_expect = np.zeros(timesteps)

initial_ti = 0
for file in os.listdir(newdir):
    ti = int(file.split('_')[file.split('_').index('ti') + 1])
    if ti > initial_ti:
        initial_ti = ti + 1
        with open(newdir + '/' + file, 'rb') as f:
            [ti, psi, projectors_left, projectors_right] = pickle.load(f)

for ti in range(initial_ti, timesteps):
    print('---')
    print(ti)
    for si in range(N):
        J_expect[ti] += bops.getOverlap(psi,
                            [tn.Node(I) for i in range(si)] + [tn.Node(np.matmul(sigma.T, sigma).reshape([1, d**2, 1]))]
                                        + [tn.Node(I) for i in range(si + 1, N)])
        for sj in range(N):
            if si < sj:
                J_expect[ti] += bops.getOverlap(psi,
                                [tn.Node(I) for i in range(si)] + [tn.Node(sigma.T.reshape([1, d**2, 1]))]
                                                + [tn.Node(I) for i in range(si + 1, sj)] + [tn.Node(sigma.reshape([1, d**2, 1]))]
                                                + [tn.Node(I) for i in range(sj +1, N)])
            elif sj < si:
                J_expect[ti] += bops.getOverlap(psi,
                                                [tn.Node(I) for i in range(sj)] + [
                                                    tn.Node(sigma.T.reshape([1, d ** 2, 1]))]
                                                + [tn.Node(I) for i in range(sj + 1, si)] + [
                                                    tn.Node(sigma.reshape([1, d ** 2, 1]))]
                                                + [tn.Node(I) for i in range(si + 1, N)])
    sigma_expect[ti] += bops.getOverlap(psi,
        [tn.Node(I) for i in range(int(N / 2))] + [tn.Node(sigma.reshape([1, d ** 2, 1]))] + [tn.Node(I) for i in range(int(N / 2) - 1)])
    sigma_T_expect[ti] += bops.getOverlap(psi,
        [tn.Node(I) for i in range(int(N / 2))] + [tn.Node(sigma.T.reshape([1, d ** 2, 1]))] + [tn.Node(I) for i in range(int(N / 2) - 1)])
    sigma_X_expect[ti] += bops.getOverlap(psi,
        [tn.Node(I) for i in range(int(N / 2))] + [tn.Node(X.reshape([1, d ** 2, 1]))] + [tn.Node(I) for i in range(int(N / 2) - 1)])
    sigma_Z_expect[ti] += bops.getOverlap(psi,
        [tn.Node(I) for i in range(int(N / 2))] + [tn.Node(Z.reshape([1, d ** 2, 1]))] + [tn.Node(I) for i in range(int(N / 2) - 1)])
    bond_dims[ti] = psi[int(len(psi)/2)].tensor.shape[0]
    if sim_method == 'tdvp':
        tdvp.tdvp_sweep(psi, L, projectors_left, projectors_right, dt / 2, max_bond_dim=bond_dim, num_of_sites=2)
    elif sim_method == 'swap':
        swap.trotter_sweep(psi, trotter_single_op, neighbor_trotter_ops, swap_op)
    if ti > 0 and ti % save_each != 1:
        old_state_filename, old_data_filename = filenames(newdir, case, N, Omega, nn_num, ti - 1, sim_method, bond_dim)
        os.remove(old_state_filename)
    state_filename, data_filename = filenames(newdir, case, N, Omega, nn_num, ti, sim_method, bond_dim)
    with open(state_filename, 'wb') as f:
        pickle.dump([ti, psi, projectors_left, projectors_right], f)

if results_to == 'plot':
    plt.plot(np.array(range(timesteps)), np.abs(J_expect), ':')
    plt.plot(np.array(range(timesteps)), np.abs(J_expect) / 2, ':')
    plt.show()