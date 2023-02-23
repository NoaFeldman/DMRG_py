import matplotlib.pyplot as plt
import numpy as np
import pickle
import basicOperations as bops
import tensornetwork as tn
import scipy.linalg as linalg
import sys

# fig = pickle.load(open('results/tdvp/steady_state_Js_N_30_k_0.1', 'rb'))
# fig.show()
# fig = pickle.load(open('results/tdvp/steady_state_JdJs_N_30_k_0.1', 'rb'))
# data0 = fig.axes[0].lines[0].get_data()
# data2 = fig.axes[0].lines[2].get_data()
# fig.show()
# fig = pickle.load(open('results/tdvp/steady_state_purity_N_30_k_0.1', 'rb'))
# fig.show()
# plt.show()
#

N = 30
k = 2 * np.pi / 10
theta = 0
mu = 3 / (2 * k * N)
Gamma = 1
gamma_1d = Gamma * mu
def get_sum_local_op_expectation_value(rho, op, factor, n_start, n_end):
    res = 0
    I = tn.Node(np.eye(2).reshape([1, 4, 1]))
    for n in range(n_start, n_end):
        res += bops.getOverlap(rho, [I] * n + [op] + [I] * (N - n - 1)) * factor**n
    return res


def get_j_expect(rho, N, sigma, phase):
    d=2
    I = np.eye(d).reshape([1, d**2, 1])
    res = 0
    for si in range(N):
        res += bops.getOverlap(rho,
                    [tn.Node(I) for i in range(si)] + [tn.Node(np.matmul(sigma.T, sigma).reshape([1, d ** 2, 1]))]
                    + [tn.Node(I) for i in range(si + 1, N)])
        for sj in range(N):
            if si < sj:
                res += bops.getOverlap(rho,
                                    [tn.Node(I) for i in range(si)] + [tn.Node(np.exp(1j * phase * si) * sigma.T.reshape([1, d ** 2, 1]))]
                                    + [tn.Node(I) for i in range(si + 1, sj)] + [
                                        tn.Node(np.exp(-1j * phase * sj) * sigma.reshape([1, d ** 2, 1]))]
                                    + [tn.Node(I) for i in range(sj + 1, N)])
            elif sj < si:
                res += bops.getOverlap(rho,
                                    [tn.Node(I) for i in range(sj)] + [
                                        tn.Node(np.exp(1j * phase * sj) * sigma.T.reshape([1, d ** 2, 1]))]
                                    + [tn.Node(I) for i in range(sj + 1, si)] + [
                                        tn.Node(np.exp(-1j * phase * si) * sigma.reshape([1, d ** 2, 1]))]
                                    + [tn.Node(I) for i in range(si + 1, N)])
    return res


def hermitian_matrix(rho):
    swap = tn.Node(np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]))
    rho_dagger = [tn.Node(bops.permute(bops.contract(node, swap, '1', '0'), [0, 2, 1]).tensor.conj()) for node in rho]
    new_rho = []
    tensor = np.zeros((1, 4, rho[0][2].dimension * 2), dtype=complex)
    tensor[0, :, :rho[0][2].dimension] = rho[0].tensor
    tensor[0, :, rho[0][2].dimension:] = rho_dagger[0].tensor
    new_rho.append(tn.Node(tensor))
    for i in range(1, len(rho) - 1):
        tensor = np.zeros((rho[i][0].dimension*2, 4, rho[i][2].dimension*2), dtype=complex)
        tensor[:rho[i][0].dimension, :, :rho[i][2].dimension] = rho[i].tensor
        tensor[rho[i][0].dimension:, :, rho[i][2].dimension:] = rho_dagger[i].tensor
        new_rho.append(tn.Node(tensor))
    tensor = np.zeros((rho[-1][0].dimension * 2, 4, 1), dtype=complex)
    tensor[:rho[-1][0].dimension, :, :] = rho[-1].tensor
    tensor[rho[-1][0].dimension:, :, :] = rho_dagger[-1].tensor
    new_rho.append(tn.Node(tensor))
    for k in range(len(new_rho) - 1, -1, -1):
        new_rho = bops.shiftWorkingSite(new_rho, k, '<<')
    for k in range(len(new_rho) - 1):
        new_rho = bops.shiftWorkingSite(new_rho, k, '>>')
    new_rho[-1].tensor /= 2
    return new_rho


d = 2
I = np.eye(d)
def get_pair_L_terms(Deltas, gammas, nearest_neighbors_num, sigma):
    A = np.kron(np.eye(d), sigma.T)
    B = np.kron(np.eye(d), sigma)
    C = np.kron(sigma.T, np.eye(d))
    D = np.kron(sigma, np.eye(d))
    return [[[(-1j * Deltas[i] - gammas[i] / 2) * A + gammas[i] * D for i in range(nearest_neighbors_num)], B],
     [[(1j * Deltas[i] - gammas[i] / 2) * C + gammas[i] * B for i in range(nearest_neighbors_num)], D],
     [[(-1j * Deltas[i] - gammas[i] / 2) * B for i in range(nearest_neighbors_num)], A],
     [[(1j * Deltas[i] - gammas[i] / 2) * D for i in range(nearest_neighbors_num)], C]]


def get_single_L_term(Omega, Gamma, sigma):
    G = -1j * Gamma / 2
    return -1j * (np.kron(np.eye(d), np.conj(Omega) * sigma + Omega * sigma.T) -
                  np.kron(Omega * sigma + np.conj(Omega) * sigma.T, np.eye(d))) \
        + Gamma * np.kron(sigma, sigma) \
        -1j * (G * np.kron(np.eye(d), np.matmul(sigma.T, sigma).T) -
               np.conj(G) * np.kron(np.matmul(sigma.T, sigma), np.eye(d)))


Omega = float(sys.argv[1])
normalized_omegas = [0.0, 0.8, 1.6, 4.0, 7.0, 10.0]
Omegas = [gamma_1d * o for o in normalized_omegas]
jdjs = np.zeros(len(Omegas))
jdjs_tst = np.zeros(len(Omegas))
sigmas = np.zeros(len(Omegas))
jxs = np.zeros(len(Omegas))
jys = np.zeros(len(Omegas))
jxs_no_phase = np.zeros(len(Omegas))
jys_no_phase = np.zeros(len(Omegas))
jdjs_no_phase = np.zeros(len(Omegas))
purities = np.zeros(len(Omegas), dtype=complex)
squeezings = np.zeros(len(Omegas), dtype=complex)
squeezings_no_phase = np.zeros(len(Omegas), dtype=complex)

X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, 1j], [-1j, 0]])
Z = np.array([[1, 0], [0, -1]])

bond = 128
model = 'Dicke_phase'
tis = [2250] #[2220, 2250, 2280, 2310, 2340, 2370, 2400, 2430, 2460, 2490]
for oi in range(len(Omegas)):
    for ti in tis:
        Omega = Omegas[oi]
        [ti, psi_1_exp, hl_1_exp, hr_1_exp, runtimes_1_exp, tes_1_exp, JdJ_1_exp, sigmaz_1_exp] = \
            pickle.load(open('results/tdvp/tst_1d/mid_state_' + model + '_N_30_Omega_' + str(Omega) + '_nn_1_ti_' + str(ti) + '_bond_' + str(bond) + '_1s_exp', 'rb'))
        dt = 1e-2
        plt.plot(np.array(range(ti)) * dt, JdJ_1_exp[:ti])
        plt.title(r'$\Omega = $' + str(normalized_omegas[oi]))
        plt.ylabel(r'$\langle J^\dagger J \rangle$')
        plt.xlabel('t')
        plt.show()
        rho = hermitian_matrix(psi_1_exp)
        jdjs_tst[oi] += get_j_expect(rho, N, np.array([[0, 0], [1, 0]]), phase=k) / N**2
        jdjs[oi] += JdJ_1_exp[ti] / N**2
        jdjs_no_phase[oi] += get_j_expect(rho, N, np.array([[0, 0], [1, 0]]), phase=0) / N**2
        sigmas[oi] += get_sum_local_op_expectation_value(rho, tn.Node(np.array([[1, 0], [0, -1]]).reshape([1, 4, 1])), 1, 5, N - 5) / (N - 10)
        j_minus = get_sum_local_op_expectation_value(rho, tn.Node(np.array([[0, 1], [0, 0]]).reshape([1, 4, 1])), np.exp(-1j * k), 5, N - 5)
        j_plus = get_sum_local_op_expectation_value(rho, tn.Node(np.array([[0, 0], [1, 0]]).reshape([1, 4, 1])), np.exp(1j * k), 5, N - 5)
        jxs[oi] += (j_minus + j_plus) / 2 / (N - 10)
        jys[oi] += (j_plus - j_minus) / 2j / (N - 10)
        j_minus_no_phase = get_sum_local_op_expectation_value(rho, tn.Node(np.array([[0, 1], [0, 0]]).reshape([1, 4, 1])), 1, 5, N - 5)
        j_plus_no_phase = get_sum_local_op_expectation_value(rho, tn.Node(np.array([[0, 0], [1, 0]]).reshape([1, 4, 1])), 1, 5, N - 5)
        print(oi, ti, j_minus_no_phase, j_plus_no_phase)
        jxs_no_phase[oi] += (j_minus_no_phase + j_plus_no_phase) / 2 / (N - 10)
        jys_no_phase[oi] += (j_plus_no_phase - j_minus_no_phase) / 2j / (N - 10)
        swap = tn.Node(np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]))
        purities[oi] += bops.getExpectationValue(psi_1_exp, [swap] * N)
    # norm = np.sqrt(jxs[oi]**2 + jys[oi]**2 + sigmas[oi]**2)
    # J_x, J_y, J_z = jxs[oi] / norm, jys[oi] / norm, sigmas[oi] / norm
    # spin_squeezing = 1
    # for jx in np.array(range(-10, 11)) * 0.1:
    #     for jy in np.array(range(-10, 11)) * 0.1:
    #         print(jx, jy)
    #         jz = (J_x * jx + J_y * jy) / J_z
    #         norm = np.sqrt(jx**2 + jy**2 + jz**2)
    #         jx_n, jy_n, jz_n = jx / norm, jy / norm, jz / norm
    #         u = linalg.expm(1j * jx_n * X + jy_n * Y + jz_n * Z)
    #         rotated_rho = [bops.permute(bops.contract(node, tn.Node(np.kron(u.T.conj(), u)), '1', '0'), [0, 2, 1])
    #                        for node in rho]
    #         curr_squeezing = get_j_expect(rotated_rho, N, np.array([[0, 0], [1, 0]]), phase=k) / N**2 - \
    #                          get_sum_local_op_expectation_value(rotated_rho, tn.Node(np.array([[1, 0], [0, -1]]).reshape([1, 4, 1])), 1, 0, N)**2 / N**2
    #         if curr_squeezing < spin_squeezing:
    #             spin_squeezing = curr_squeezing
    #         print(curr_squeezing, get_sum_local_op_expectation_value(rotated_rho, tn.Node(np.array([[1, 0], [0, -1]]).reshape([1, 4, 1])), 1, 0, N))
    # squeezings[oi] = spin_squeezing
    # J_x, J_y, J_z = jxs_no_phase[oi] / norm, jys_no_phase[oi] / norm, sigmas[oi] / norm
    # spin_squeezing = 1
    # for jx in np.array(range(-10, 11)) * 0.1:
    #     for jy in np.array(range(-10, 11)) * 0.1:
    #         print(jx, jy)
    #         jz = (J_x * jx + J_y * jy) / J_z
    #         norm = np.sqrt(jx**2 + jy**2 + jz**2)
    #         jx, jy, jz = jx / norm, jy / norm, jz / norm
    #         u = linalg.expm(1j * jx * X + jy * Y + jz * Z)
    #         rotated_rho = [bops.permute(bops.contract(node, tn.Node(np.kron(u, u.T.conj())), '1', '0'), [0, 2, 1])
    #                        for node in rho]
    #         curr_squeezing = get_j_expect(rotated_rho, N, np.array([[0, 0], [1, 0]]), phase=0) - \
    #                          get_sum_local_op_expectation_value(rotated_rho, tn.Node(np.array([[1, 0], [0, -1]]).reshape([1, 4, 1])), 1, 0, N)
    #         if curr_squeezing < spin_squeezing:
    #             spin_squeezing = curr_squeezing
    # squeezings_no_phase[oi] = spin_squeezing


fig, ax = plt.subplots()
ax.plot(normalized_omegas, sigmas / len(tis))
ax.plot(normalized_omegas, jxs / len(tis))
ax.plot(normalized_omegas, jys / len(tis))
ax.plot(normalized_omegas, jxs_no_phase / len(tis))
ax.plot(normalized_omegas, jys_no_phase / len(tis))
ax.legend([r'$\sum_{n=5}^{N-5} \langle \sigma^z_i \rangle / (N-10)$', r'$\langle J_x \rangle$', r'$\langle J_y \rangle$',
            r'$\langle J^\prime_x \rangle$ (no phase)', r'$\langle J^\prime_y \rangle$ (no phase)'], fontsize=16)
ax.set_xlabel(r'$\Omega/\gamma_1d$', fontsize=16)
ax.tick_params(axis="x", labelsize=16)
ax.tick_params(axis="y", labelsize=16)
pickle.dump(fig, open('results/tdvp/steady_state_Js_N_' + str(N) + '_k_' + str(np.round(k / 2 / np.pi, 2)), 'wb'))
plt.show()

fig, ax = plt.subplots()
ax.plot(normalized_omegas, jdjs / len(tis))
ax.plot(normalized_omegas, jdjs_tst / len(tis), '--')
ax.plot(normalized_omegas, jdjs_no_phase / len(tis))
plt.legend([r'$\langle J^\dagger J \rangle / N^2$', r'$\langle J^{\prime\dagger} J^\prime \rangle / N^2$'], fontsize=16)
ax.set_xlabel(r'$\Omega/\gamma_1d$', fontsize=16)
ax.tick_params(axis="x", labelsize=16)
ax.tick_params(axis="y", labelsize=16)
pickle.dump(fig, open('results/tdvp/steady_state_JdJs_N_' + str(N) + '_k_' + str(np.round(k / 2 / np.pi, 2)), 'wb'))
plt.show()

fig, ax = plt.subplots()
ax.plot(normalized_omegas, squeezings * N / 4)
ax.plot(normalized_omegas, squeezings * N / 4)
plt.legend([r'squeezing', r'squeezing$^\prime$'], fontsize=16)
ax.set_xlabel(r'$\Omega/\gamma_1d$', fontsize=16)
ax.tick_params(axis="x", labelsize=16)
ax.tick_params(axis="y", labelsize=16)
pickle.dump(fig, open('results/tdvp/steady_state_JdJs_N_' + str(N) + '_k_' + str(np.round(k / 2 / np.pi, 2)), 'wb'))
plt.show()

fig, ax = plt.subplots()
ax.plot(normalized_omegas, purities / len(tis))
ax.set_xlabel(r'$\Omega/\gamma_1d$', fontsize=16)
ax.set_ylabel(r'$p_2$', fontsize=16)
ax.tick_params(axis="x", labelsize=16)
ax.tick_params(axis="y", labelsize=16)
pickle.dump(fig, open('results/tdvp/steady_state_purity_N_' + str(N) + '_k_' + str(np.round(k / 2 / np.pi, 2)), 'wb'))
plt.show()
