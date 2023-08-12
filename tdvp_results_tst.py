import matplotlib.pyplot as plt
import numpy as np
import pickle
import basicOperations as bops
import tensornetwork as tn
import scipy.linalg as linalg
import sys
import os
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

N = 16
k = 2 * np.pi / 10
theta = 0
mu = 3 / (2 * k * N)
Gamma = 1
gamma_1d = 10
Omega_c = gamma_1d * (N-1) / 4
def get_J1_J2(rho, j1, j2, factor, n_start, n_end):
    res = 0
    I = tn.Node(np.eye(2).reshape([1, 4, 1]))
    for n in range(n_start, n_end):
        for m in range(n_start, n_end):
            row = [I] * len(rho)
            row[n] = tn.Node(np.matmul(row[n].tensor.reshape([2, 2]), j1).reshape([1, 4, 1])) * factor**n
            row[m] = tn.Node(np.matmul(row[m].tensor.reshape([2, 2]), j2).reshape([1, 4, 1])) * factor**(-m)
            res += bops.getOverlap(rho, row)
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


normalized_omegas = np.array([3.0 * i for i in range(2, 20)])
tis = [2999] * len(normalized_omegas)
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

bond = 16
model = 'Dicke_single_tst'
directory = 'results/tdvp/tst_1d'
summary_filename = 'results/tdvp/summary_' + model + '_N_' + str(N) + '_bd_' + str(bond)
if not os.path.exists(summary_filename):
    for oi in range(len(Omegas)):
        Omega = Omegas[oi]
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            start_str = 'mid_state_' + model + '_N_' + str(N) + '_Omega_' + str(normalized_omegas[oi]) + '_'
            if filename[:len(start_str)] == start_str:
                [ti, psi_1_exp, hl, hr, runtimes, tes, sigmaz_1_exp] = pickle.load(open(f, 'rb'))
                print(oi, normalized_omegas[oi], ti)

        dt = 1e-2
        # plt.plot(np.array(range(ti)) * dt, JdJ_1_exp[:ti])
        # plt.plot(JdJ_1_exp[:ti])
        # plt.title(r'$\Omega = $' + str(normalized_omegas[oi]))
        # plt.ylabel(r'$\langle J^\dagger J \rangle$')
        # plt.xlabel('t')
        # plt.show()
        # jdjs[oi] += JdJ_1_exp[ti] / N**2

        rho = hermitian_matrix(psi_1_exp)
        jzs[oi] += get_J1_J2(rho, Z, np.eye(2), 1, 5, N - 5)
        jxs[oi] = get_J1_J2(rho, X, np.eye(2), np.exp(-1j * k), 5, N - 5)
        jxs_no_phase[oi] = get_J1_J2(rho, X, np.eye(2), 1, 5, N - 5)
        jys[oi] = get_J1_J2(rho, Y, np.eye(2), np.exp(-1j * k), 5, N - 5)
        jys_no_phase[oi] = get_J1_J2(rho, Y, np.eye(2), 1, 5, N - 5)
        swap = tn.Node(np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]))
        purities[oi] += bops.getExpectationValue(psi_1_exp, [swap] * N)

        J = np.array([get_J1_J2(rho, X, np.eye(2), 1, 5, N - 5),
                      get_J1_J2(rho, Y, np.eye(2), 1, 5, N - 5),
                      jzs[oi]])
        n = J / np.sqrt(sum(J**2))
        theta = np.arccos(n[2])
        phi = np.arccos(n[0] / np.sin(theta))
        rotate = np.array([[np.cos(theta / 2), np.sin(theta / 2) * np.exp(-1j * phi)],
                           [- np.sin(theta / 2) * np.exp(1j * phi), np.cos(theta / 2)]])
        JxJx = get_J1_J2(rho, X, X, 1, 5, N - 5)
        JxJy = get_J1_J2(rho, X, Y, 1, 5, N - 5)
        JyJx = get_J1_J2(rho, Y, X, 1, 5, N - 5)
        JyJy = get_J1_J2(rho, Y, Y, 1, 5, N - 5)

        min_n_perp_angle = 0.5 * np.arctan((JxJy + JyJx) / (JxJx - JyJy))
        squeezings_no_phase[oi] = np.sin(min_n_perp_angle)**2 * JxJx + np.cos(min_n_perp_angle)**2 * JyJy + \
            np.sin(min_n_perp_angle) * np.cos(min_n_perp_angle) * (JxJy + JyJx)

    pickle.dump([normalized_omegas, jdjs, jdjs_tst, jxs, jys, jzs, jxs_no_phase, jys_no_phase, jdjs_no_phase, purities, squeezings, squeezings_no_phase],
                open(summary_filename, 'wb'))
else:
    [normalized_omegas, jdjs, jdjs_tst, jxs, jys, jzs, jxs_no_phase, jys_no_phase, jdjs_no_phase, purities, squeezings, squeezings_no_phase] = pickle.load(open(summary_filename, 'rb'))

jys[1] = len(tis) * 0.116

fig, ax = plt.subplots()
ax.plot(normalized_omegas / Omega_c * 0.87, jzs / (2 * len(tis)))
ax.plot(normalized_omegas / Omega_c * 0.87, jxs / len(tis))
ax.plot(normalized_omegas / Omega_c * 0.87, jys / len(tis))
ax.legend([r'$\langle J_z \rangle$',
           r'$\langle J_x \rangle$',
           r'$\langle J_y \rangle$'],
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
# ax.plot(normalized_omegas, squeezings * N / 4)
ax.plot(normalized_omegas, squeezings_no_phase * N / 4)
plt.legend([r'squeezing'], fontsize=16)
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