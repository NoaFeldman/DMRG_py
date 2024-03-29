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


def get_pair_L_terms(Deltas, gammas, nearest_neighbors_num, sigma):
    A = np.kron(np.eye(d), sigma.T)
    B = np.kron(np.eye(d), sigma)
    C = np.kron(sigma.T, np.eye(d))
    D = np.kron(sigma, np.eye(d))
    return [[[(-1j * Deltas[i] - gammas[i] / 2) * A + gammas[i] * D for i in range(nearest_neighbors_num)], B],
     [[(1j * Deltas[i] - gammas[i] / 2) * C + gammas[i] * B for i in range(nearest_neighbors_num)], D],
     [[(-1j * Deltas[i] - gammas[i] / 2) * B for i in range(nearest_neighbors_num)], A],
     [[(1j * Deltas[i] - gammas[i] / 2) * D for i in range(nearest_neighbors_num)], C]]


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



def filenames(newdir, case, N, Omega, ti, bond_dim):
    state_filename = newdir + '/mid_state_' + case + '_N_' + str(N) \
        + '_Omega_' + str(Omega) + '_ti_' + str(ti) + '_bond_' + str(bond_dim)
    data_filename = newdir + '/tdvp_N_' + str(N) \
                      + '_Omega_' + str(Omega) + '_bond_' + str(bond_dim)
    return state_filename, data_filename


def get_sigma_z_expect(rho, N):
    res = 0
    for si in range(N):
        res += bops.getOverlap(rho,
            [tn.Node(I) for i in range(si)] + [
               tn.Node(np.diag([1, -1]).reshape([1, d ** 2, 1]))]
            + [tn.Node(I) for i in range(si + 1, N)])
    return res / N


def get_j_expect(rho, N, sigma):
    res = 0
    for si in range(N):
        res += bops.getOverlap(rho,
                    [tn.Node(I) for i in range(si)] + [tn.Node(np.matmul(sigma.T, sigma).reshape([1, d ** 2, 1]))]
                    + [tn.Node(I) for i in range(si + 1, N)])
        for sj in range(si + 1, N):
            res += bops.getOverlap(rho,
                                   [tn.Node(I) for i in range(si)] + [
                                       tn.Node(np.exp(-1j * k * si) * sigma.T.reshape([1, d ** 2, 1]))]
                                   + [tn.Node(I) for i in range(si + 1, sj)] + [
                                       tn.Node(np.exp(1j * k * sj) * sigma.reshape([1, d ** 2, 1]))]
                                   + [tn.Node(I) for i in range(sj + 1, N)])
            res += bops.getOverlap(rho,
                                   [tn.Node(I) for i in range(si)] + [
                                       tn.Node(np.exp(1j * k * si) * sigma.reshape([1, d ** 2, 1]))]
                                   + [tn.Node(I) for i in range(si + 1, sj)] + [
                                       tn.Node(np.exp(-1j * k * sj) * sigma.T.reshape([1, d ** 2, 1]))]
                                   + [tn.Node(I) for i in range(sj + 1, N)])
    return res


def get_single_L_term(Omega, Gamma, sigma, is_single, gamma_1d=0):
    result = -1j * (np.kron(np.eye(d), np.conj(Omega) * sigma + Omega * sigma.T) -
                      np.kron(Omega * sigma + np.conj(Omega) * sigma.T, np.eye(d)))
    single_dissipation = (Gamma + gamma_1d) if is_single else gamma_1d
    result += single_dissipation * np.kron(sigma, sigma)
    result -= single_dissipation * (np.kron(np.eye(d), np.matmul(sigma.T, sigma)) +
                         np.kron(np.matmul(sigma.T, sigma).T, np.eye(d))) / 2
    return result

def get_photon_green_L_exp(n, Omega, Gamma, gamma_1d, k, theta, sigma, is_Delta=True, is_chiral=False, is_single=True, is_same_site=False, with_a=False):
    d = 2
    if is_same_site:
        phase = 1
    else:
        phase = np.exp(1j * k)
    Ss = [get_single_L_term(Omega * phase**i, Gamma, sigma, is_single, gamma_1d=gamma_1d) for i in range(n)]
    # Ss = [get_single_L_term(Omega, Gamma, sigma, is_single, gamma_1d=gamma_1d) for i in range(n)]

    interacting_terms = [
        [gamma_1d * np.kron(sigma, I), phase**(-1) * np.kron(I, I), phase**(-1) * np.kron(I, sigma)],
        [gamma_1d * np.kron(I, sigma), phase**(1) * np.kron(I, I), phase**(1) * np.kron(sigma, I)],
        [-gamma_1d / 2 * np.kron(I, sigma), phase ** (-1) * np.kron(I, I), phase ** (-1) * np.kron(I, sigma.T)],
        [-gamma_1d / 2 * np.kron(sigma.T, I), phase ** (-1) * np.kron(I, I), phase ** (-1) * np.kron(sigma, I)],
        [-gamma_1d / 2 * np.kron(I, sigma.T), phase ** (1) * np.kron(I, I), phase ** (1) * np.kron(I, sigma)],
        [-gamma_1d / 2 * np.kron(sigma, I), phase ** (1) * np.kron(I, I), phase ** (1) * np.kron(sigma.T, I)],
    ]

    if not is_chiral:
        interacting_terms += [
            [gamma_1d / 2 * np.kron(sigma, I), phase * np.kron(I, I), phase * np.kron(I, sigma)],
            [gamma_1d / 2 * np.kron(I, sigma), phase * np.kron(I, I), phase * np.kron(sigma, I)],
        ]
        # TODO verify the terms below!
        interacting_terms += [
            [-gamma_1d / 2 * np.kron(I, sigma), phase ** (-1) * np.kron(I, I), phase ** (-1) * np.kron(I, sigma.T)],
            [-gamma_1d / 2 * np.kron(sigma.T, I), phase ** (-1) * np.kron(I, I), phase ** (-1) * np.kron(sigma, I)],
            [-gamma_1d / 2 * np.kron(I, sigma.T), phase ** (1) * np.kron(I, I), phase ** (1) * np.kron(I, sigma)],
            [-gamma_1d / 2 * np.kron(sigma, I), phase ** (1) * np.kron(I, I), phase ** (1) * np.kron(sigma.T, I)],
        ]
    if is_Delta:
        if is_chiral:
            return None
        else:
            interacting_terms += [
                [- gamma_1d / 2 * np.kron(I, sigma.T), phase * np.kron(I, I), phase * np.kron(I, sigma)],
                         [- gamma_1d / 2 * np.kron(I, sigma), phase * np.kron(I, I), phase * np.kron(I, sigma.T)],
                         [- gamma_1d / 2 * np.kron(sigma.T, I), phase**(-1) * np.kron(I, I), phase**(-1) * np.kron(sigma, I)],
                         [- gamma_1d / 2 * np.kron(sigma, I), phase**(-1) * np.kron(I, I), phase**(-1) * np.kron(sigma.T, I)],
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


d = 2
Gamma = 1
sigma = np.array([[0, 0], [1, 0]])
I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Z = np.diag([1, -1])

N = int(sys.argv[1])
k = 2 * np.pi / 10
theta = 0
case = sys.argv[3]
mu = 3 / (2 * k * N)
gamma_1d = 10 # Gamma * mu
Omega = float(sys.argv[2])
outdir = sys.argv[4]
timesteps = int(sys.argv[5])
dt = 1e-4
save_each = 30
results_to = sys.argv[6]
bond_dim = int(sys.argv[7])

newdir = outdir + '/' + case + '_N_' + str(N) + '_Omega_' + str(Omega) + '_nn_' + '_bd_' + str(bond_dim)
Omega *= Gamma
try:
    os.mkdir(newdir)
except FileExistsError:
    pass

if results_to == 'plot':
    import matplotlib.pyplot as plt

Gamma = 1
gamma_1d = 10
if case == 'kernel_1d':
    L_exp = get_photon_green_L_exp(N, Omega, Gamma, gamma_1d, k, theta, sigma, is_Delta=True, is_single=True, is_chiral=False, is_same_site=False)
elif case == 'Dicke_phase':
    L_exp = get_photon_green_L_exp(N, Omega, Gamma, gamma_1d, k, theta, sigma, is_Delta=False, is_single=False, is_chiral=True, is_same_site=False)
elif case == 'Dicke_single':
    L_exp = get_photon_green_L_exp(N, Omega, Gamma, gamma_1d, k, theta, sigma, is_Delta=False, is_single=True, is_chiral=True, is_same_site=False)
elif case == 'Dicke_single_tst':
    L_exp = get_photon_green_L_exp(N, Omega, Gamma, gamma_1d, k, theta, sigma, is_Delta=False, is_single=True, is_chiral=True, is_same_site=True)


sigmas = [np.kron(np.eye(d ** i), np.kron(sigma, np.eye(d ** (N - i - 1)))) for i in range(N)]
if N <= 6:
    print('starting exact tests')
    mpo = bops.contract(L_exp[0], L_exp[1], '3', '2')
    for i in range(2, N):
        mpo = bops.contract(mpo, L_exp[i], [1 + 2 * i], '2')
    L_exp_mat = mpo.tensor.reshape([d] * 4 * N).transpose([0 + i * 4 for i in range(N)] +
                                                          [1 + i * 4 for i in range(N)] +
                                                          [2 + i * 4 for i in range(N)] +
                                                          [3 + i * 4 for i in range(N)]).reshape([d**(2 * N)] * 2).T
    L_mat = np.zeros((4**N, 4**N), dtype=complex)
    L_mat_test = np.zeros((4**N, 4**N), dtype=complex)

    S = np.zeros((2 ** N, 2 ** N))
    SX = np.zeros((2 ** N, 2 ** N))
    Zs = np.zeros([2 ** N, 2 ** N], dtype=complex)
    for i in range(N):
        S += np.kron(np.eye(2 ** i), np.kron(sigma, np.eye(2 ** (N - i - 1))))
        SX += np.kron(np.eye(2 ** i), np.kron(sigma + sigma.T, np.eye(2 ** (N - i - 1))))
        Zs += np.kron(np.eye(2 ** i), np.kron(np.diag([1, -1]), np.eye(2 ** (N - i - 1))))
    H = Omega * SX

    L_mat = -1j * (np.kron(np.eye(2**N), H) - np.kron(H.T, np.eye(2**N)))
    L_mat += gamma_1d / 2 * (2 * np.kron(S.conj(), S)
                             - np.kron(np.matmul(S.T.conj(), S).T, np.eye(2**N))
                             - np.kron(np.eye(2**N), np.matmul(S.T.conj(), S)))
    L_mat_test = -1j * (np.kron(np.eye(2**N), H) - np.kron(H.T, np.eye(2**N)))
    for i in range(N):
        L_mat += (Gamma) / 2 * (2 * np.kron(sigmas[i].conj(), sigmas[i])
                                - np.kron(np.matmul(sigmas[i].T.conj(), sigmas[i]).T, np.eye(2**N))
                                - np.kron(np.eye(2**N), np.matmul(sigmas[i].T.conj(), sigmas[i])))
        L_mat_test += (Gamma) / 2 * (2 * np.kron(sigmas[i].conj(), sigmas[i])
                                - np.kron(np.matmul(sigmas[i].T.conj(), sigmas[i]).T, np.eye(2**N))
                                - np.kron(np.eye(2**N), np.matmul(sigmas[i].T.conj(), sigmas[i])))
        for j in range(N):
            L_mat_test += gamma_1d * np.kron(sigmas[i].conj(), sigmas[j]) \
                - gamma_1d / 2 * np.kron(np.matmul(sigmas[j].T, sigmas[i]).T, np.eye(2**N)) \
                - gamma_1d / 2 * np.kron(np.eye(2**N), np.matmul(sigmas[j].T, sigmas[i]))
    U = linalg.expm(L_mat * dt)

    psi_exact = np.array([0] * (4**N - 1) + [1], dtype=complex)
    rho = psi_exact.reshape([2 ** N] * 2)
    Zs = np.zeros([2**N, 2**N], dtype=complex)
    for i in range(N):
        Zs += np.kron(np.eye(2**i), np.kron(np.diag([1, -1]), np.eye(2**(N - i - 1))))
    Js = np.zeros(timesteps, dtype=complex)
    for ti in range(timesteps):
        print(ti)
        psi_exact = np.matmul(U, psi_exact)
        rho = psi_exact.reshape([2**N] * 2)
        print(np.amax(np.abs(rho - rho.T.conj())))
        psi_exact /= rho.trace()
        Js[ti] = np.matmul(rho / rho.trace(), Zs).trace()
    print(Js[0])
    import matplotlib.pyplot as plt
    plt.plot(dt * np.array(range(timesteps)), np.real(Js) / N)
    # plt.show()

if N > 6 and N < 16:
    if case == 'Dicke_single':
        psi_exact = np.array([0] * (4 ** N - 1) + [1], dtype=complex)
        rho = psi_exact.reshape([2 ** N] * 2)
        Js = np.zeros(timesteps, dtype=complex)
        S = np.zeros((2**N, 2**N))
        SX = np.zeros((2**N, 2**N))
        Zs = np.zeros([2 ** N, 2 ** N], dtype=complex)
        for i in range(N):
            S += np.kron(np.eye(2**i), np.kron(sigma, np.eye(2**(N - i - 1))))
            SX += np.kron(np.eye(2**i), np.kron(sigma + sigma.T, np.eye(2**(N - i - 1))))
            Zs += np.kron(np.eye(2 ** i), np.kron(np.diag([1, -1]), np.eye(2 ** (N - i - 1))))
        H = Omega * SX
        for ti in range(timesteps):
            print(ti)
            drho_dt = -1j * (np.matmul(H, rho) - np.matmul(rho, H))
            drho_dt += gamma_1d / 2 * (2 * np.matmul(S, np.matmul(rho, S.T.conj()))
                                - np.matmul(rho, np.matmul(S.T.conj(), S)) - np.matmul(np.matmul(S.T.conj(), S), rho))
            for i in range(N):
                drho_dt += Gamma / 2 * (2 * np.matmul(sigmas[i], np.matmul(rho, sigmas[i].T.conj())) -
                                        np.matmul(rho, np.matmul(sigmas[i].T.conj(), sigmas[i])) -
                                        np.matmul(np.matmul(sigmas[i].T.conj(), sigmas[i]), rho))
            rho += dt * drho_dt
            rho /= rho.trace()
            Js[ti] = np.matmul(rho / rho.trace(), Zs).trace()
        import matplotlib.pyplot as plt
        pickle.dump(Js, open(newdir + '/exact', 'wb'))
        plt.plot(dt * np.array(range(timesteps)), np.real(Js) / N)
        # plt.show()


I = np.eye(2).reshape([1, d**2, 1])
J_expect = np.zeros(timesteps, dtype=complex)
bond_dims = np.zeros(timesteps, dtype=complex)

psi = [tn.Node(np.array([0., 0., 0., 1. + 0j]).reshape([1, d**2, 1])) for n in range(N)]

psi_1_exp = bops.copyState(psi)
# projectors_left, projectors_right = tdvp.get_initial_projectors(psi, L)
hl_1_exp, hr_1_exp = tdvp.get_initial_projectors(psi, L_exp)

runtimes_1_exp = np.zeros(timesteps)

tes_1_exp = np.zeros(timesteps)
sigmaz_1_exp = np.zeros(timesteps)

if len(sys.argv) == 9:
    filename = sys.argv[8]
    data = pickle.load(open(filename, 'rb'))
    [ti, psi_1_exp, hl_1_exp, hr_1_exp] = data[:4]

initial_ti = 0
for file in os.listdir(newdir):
    if file[-7:] == '_1s_exp':
        ti = int(file.split('_')[file.split('_').index('ti') + 1])
        if ti + 1 > initial_ti:
            try:
                data = pickle.load(open(newdir + '/' + file, 'rb'))
                initial_ti = ti + 1
                [ti, psi_1_exp, hl_1_exp, hr_1_exp] = data[:4]
                runtimes_1_exp[:len(data[4])] = data[4]
                if len(data) > 5: tes_1_exp[:len(data[5])] = data[5]
                if len(data) >= 7:
                    sigmaz_1_exp[:ti + 1] = data[6][:ti + 1]
            except Exception:
                continue

curr_bond_dim = bond_dim # 8 * 2**(int(initial_ti / 100))
for ti in range(initial_ti, timesteps):
    print('---')
    get_sigma_z_expect(psi_1_exp, N)
    print(ti)
    if ti % 100 == 0 and curr_bond_dim < bond_dim:
        curr_bond_dim *= 2
    if ti > 0 and ti % save_each != 1:
        old_state_filename, old_data_filename = filenames(newdir, case, N, Omega, ti - 1, bond_dim)
        os.remove(old_state_filename + '_1s_exp')
    tstart = time.time()
    tes_1_exp[ti] = tdvp.tdvp_sweep(psi_1_exp, L_exp, hl_1_exp, hr_1_exp, dt / 2, max_bond_dim=curr_bond_dim, num_of_sites=2)
    sigmaz_1_exp[ti] = get_sigma_z_expect(psi_1_exp, N)
    print(sigmaz_1_exp[ti])
    tf = time.time()
    runtimes_1_exp[ti] = tf - tstart
    print('times = ' + str([runtimes_1_exp[ti]]) + ', te = ' + str(tes_1_exp[ti]))
    state_filename, data_filename = filenames(newdir, case, N, Omega, ti, bond_dim)
    with open(state_filename + '_1s_exp', 'wb') as f:
        pickle.dump([ti, psi_1_exp, hl_1_exp, hr_1_exp, runtimes_1_exp, tes_1_exp, sigmaz_1_exp], f)

plt.plot(dt * np.array(range(len(sigmaz_1_exp))), sigmaz_1_exp, '--')
plt.savefig('tst_dt_' + str(dt) + case + '_Omega_' + str(Omega) + '_bd_' + str(bond_dim) + '.pdf')
plt.show()