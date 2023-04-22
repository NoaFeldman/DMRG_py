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
                    [tn.Node(I) for i in range(si)] + [tn.Node(np.matmul(sigma.conj().T, sigma).reshape([1, d ** 2, 1]))]
                    + [tn.Node(I) for i in range(si + 1, N)])
        for sj in range(N):
            if si < sj:
                res += bops.getOverlap(rho,
                                    [tn.Node(I) for i in range(si)] + [tn.Node(np.exp(1j * phase * si) * sigma.conj().T.reshape([1, d ** 2, 1]))]
                                    + [tn.Node(I) for i in range(si + 1, sj)] + [
                                        tn.Node(np.exp(-1j * phase * sj) * sigma.reshape([1, d ** 2, 1]))]
                                    + [tn.Node(I) for i in range(sj + 1, N)])
            elif sj < si:
                res += bops.getOverlap(rho,
                                    [tn.Node(I) for i in range(sj)] + [
                                        tn.Node(np.exp(1j * phase * sj) * sigma.conj().T.reshape([1, d ** 2, 1]))]
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


normalized_omegas = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6] #[np.round(0.01 * i,  3) for i in range(1, 10)] + [0.1, 0.2, 0.3, 0.4]
tis = [5999] * len(normalized_omegas)
Omegas = [gamma_1d * o for o in normalized_omegas]
jdjs = np.zeros(len(Omegas), dtype=complex)
jdjs_tst = np.zeros(len(Omegas), dtype=complex)
jxs = np.zeros(len(Omegas), dtype=complex)
jys = np.zeros(len(Omegas), dtype=complex)
jzs = np.zeros(len(Omegas), dtype=complex)
jxs_no_phase = np.zeros(len(Omegas), dtype=complex)
jys_no_phase = np.zeros(len(Omegas), dtype=complex)
jdjs_no_phase = np.zeros(len(Omegas), dtype=complex)
purities = np.zeros(len(Omegas), dtype=complex)
squeezings = np.zeros(len(Omegas), dtype=complex)
squeezings_no_phase = np.zeros(len(Omegas), dtype=complex)

X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, 1j], [-1j, 0]])
Z = np.array([[1, 0], [0, -1]])

bond = 128
model = 'Dicke_single'
for oi in range(len(Omegas)):
    ti = tis[oi]
    Omega = Omegas[oi]
    print('--', normalized_omegas[oi])
    [ti, psi_1_exp, hl_1_exp, hr_1_exp, runtimes_1_exp, tes_1_exp, JdJ_1_exp, sigmaz_1_exp] = \
        pickle.load(open('results/tdvp/tst_1d/mid_state_' + model + '_N_30_Omega_' + str(normalized_omegas[oi]) + '_nn_1_ti_' + str(ti) + '_bond_' + str(bond) + '_1s_exp', 'rb'))
    dt = 1e-2
    # plt.plot(np.array(range(ti)) * dt, JdJ_1_exp[:ti])
    # plt.plot(JdJ_1_exp[:ti])
    # plt.title(r'$\Omega = $' + str(normalized_omegas[oi]))
    # plt.ylabel(r'$\langle J^\dagger J \rangle$')
    # plt.xlabel('t')
    # plt.show()
    jdjs[oi] += JdJ_1_exp[ti] / N**2

    rho = hermitian_matrix(psi_1_exp)
    jzs[oi] += get_sum_local_op_expectation_value(rho, tn.Node(Z.reshape([1, 4, 1])), 1, 5, N - 5)
    jxs[oi] = get_sum_local_op_expectation_value(rho, tn.Node(X.reshape([1, 4, 1])), np.exp(-1j * k), 5, N - 5)
    jxs_no_phase[oi] = get_sum_local_op_expectation_value(rho, tn.Node(X.reshape([1, 4, 1])), 1, 5, N - 5)
    jys[oi] = get_sum_local_op_expectation_value(rho, tn.Node(Y.reshape([1, 4, 1])), np.exp(-1j * k), 5, N - 5)
    jys_no_phase[oi] = get_sum_local_op_expectation_value(rho, tn.Node(Y.reshape([1, 4, 1])), 1, 5, N - 5)
    swap = tn.Node(np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]))
    purities[oi] += bops.getExpectationValue(psi_1_exp, [swap] * N)

    # J = np.array([get_sum_local_op_expectation_value(rho, tn.Node(X.reshape([1, 4, 1])), 1, 5, N - 5),
    #               get_sum_local_op_expectation_value(rho, tn.Node(Y.reshape([1, 4, 1])), 1, 5, N - 5),
    #               jzs[oi]])
    # n = J / np.sqrt(sum(J**2))
    # theta = np.arccos(n[2])
    # phi = np.arccos(n[0] / np.sin(theta))
    # rotate = np.array([[np.cos(theta / 2), np.sin(theta / 2) * np.exp(-1j * phi)],
    #                    [- np.sin(theta / 2) * np.exp(1j * phi), np.cos(theta / 2)]])
    #
    # sqr = get_j_expect(rho, N, np.matmul(rotate.conj().T, np.matmul(Z, rotate)), 0)
    # sing = get_sum_local_op_expectation_value(rho,
    #                                           tn.Node(np.matmul(rotate.conj().T, np.matmul(Z, rotate)).reshape(
    #                                               [1, 4, 1])), 1, 0, N)
    # xi = sqr - np.abs(sing) ** 2
    # print(xi, sqr, sing)
    #
    # squeeze_min = 100
    # n_perp_steps = 10
    # for x in [si / n_perp_steps for si in range(n_perp_steps)]:
    #     J_op = np.array([[0, x - 1j * np.sqrt(1 - x**2)], [x + 1j * np.sqrt(1 - x**2), 0]])
    #     sqr = get_j_expect(rho, N, np.matmul(rotate.conj().T, np.matmul(J_op, rotate)), 0)
    #     sing = get_sum_local_op_expectation_value(rho,
    #             tn.Node(np.matmul(rotate.conj().T, np.matmul(J_op, rotate)).reshape([1, 4, 1])), 1, 0, N)
    #     xi = (sqr - np.abs(sing)**2) / N
    #     print(xi, sqr, sing)
    #     if xi < squeeze_min:
    #         squeeze_min = xi
    # squeezings_no_phase[oi] = squeeze_min



fig, ax = plt.subplots()
ax.plot(normalized_omegas, jzs / len(tis) - jzs[0] / len(tis))
ax.plot(normalized_omegas, jxs / len(tis) - jxs[0] / len(tis))
ax.plot(normalized_omegas, jys / len(tis) - jys[0] / len(tis))
ax.plot(normalized_omegas, jxs_no_phase / len(tis) - jxs_no_phase[0] / len(tis), '--')
ax.plot(normalized_omegas, jys_no_phase / len(tis) - jys_no_phase[0] / len(tis), '--')
ax.legend([r'$\sum_{n=5}^{N-5} \langle \sigma^z_i \rangle / (N-10) - $' + str(np.real(np.round(jzs[0] / len(tis), 2))),
           r'$\langle J_x \rangle - $' + str(np.real(np.round(jxs[0] / len(tis), 2))),
           r'$\langle J_y \rangle - $' + str(np.real(np.round(jys[0] / len(tis), 2))),
           r'$\langle J^\prime_x \rangle$ (no phase) - ' + str(np.real(np.round(jxs_no_phase[0] / len(tis), 2))),
           r'$\langle J^\prime_y \rangle$ (no phase) - ' + str(np.real(np.round(jys_no_phase[0] / len(tis), 2)))],
          fontsize=16)
ax.set_xlabel(r'$\Omega/\gamma_1d$', fontsize=16)
ax.tick_params(axis="x", labelsize=16)
ax.tick_params(axis="y", labelsize=16)
pickle.dump(fig, open('results/tdvp/steady_state_Js_N_' + str(N) + '_k_' + str(np.round(k / 2 / np.pi, 2)), 'wb'))
plt.show()

fig, ax = plt.subplots()
ax.plot(normalized_omegas, jdjs / len(tis))
plt.legend([r'$\langle J^\dagger J \rangle / N^2$'], fontsize=16)
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
