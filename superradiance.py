import numpy as np
import pickle
import tdvp
import sys
import os
import tensornetwork as tn
import basicOperations as bops
import scipy.linalg as linalg


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
    I = np.eye(4)
    return L


d = 2
Gamma = 1
sigma = np.array([[0, 0], [1, 0]])
I = np.eye(2).reshape([1, d ** 2, 1])

do_run = int(sys.argv[1])
if do_run:
    N = int(sys.argv[2])
    k = 2 * np.pi / 10
    theta = 0
    nn_num = int(sys.argv[3])
    Omega = float(sys.argv[4]) / Gamma
    case = sys.argv[5]
    outdir = sys.argv[6]
    timesteps = int(sys.argv[7])

    newdir = outdir + '/' + case + '_N_' + str(N) + '_Omega_' + str(Omega) + '_nn_' + str(nn_num) + '_timesteps_' + str(timesteps)
    try:
        os.mkdir(newdir)
    except FileExistsError:
        pass

    T = 4
    dt = T / timesteps
    save_each = 10
    results_to = sys.argv[8]

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
        evolver = linalg.expm(L_mat * dt)
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
            # J_expect_half_exact[ti] = np.abs(np.trace(np.matmul(JdJ, rho_vec_half_exact.reshape([d**N, d**N]))))
            projector = np.zeros((len(rho_vec), len(rho_vec)), dtype=complex)
            rho_mps = bops.vector_to_mps(rho_vec.reshape([d]*(2 * N))\
                                         .transpose(np.array([[i, i + N] for i in range(N)]).reshape([2 * N])), d**2, N)
            for si in range(N - 1, 0, -1):
                if si > 1:
                    curr_l = tn.Node(np.eye(1))
                    for sli in range(si - 1):
                        curr_l = bops.contract(curr_l, rho_mps[sli], [1 + sli], '0')
                    projector_l = bops.contract(curr_l, curr_l, [si, '*'], [si])
                    curr_projector = np.kron(projector_l.tensor.reshape([d**(2*si - 2), d**(2 * si - 2)]),
                                             np.eye(d**4))
                else:
                    curr_projector = np.eye(d**4)
                if si < N - 1:
                    curr_r = tn.Node(np.eye(rho_mps[si + 1][0].dimension))
                    for sri in range(si + 1, N):
                        curr_r = bops.contract(curr_r, rho_mps[sri], [sri - si], '0')
                    projector_r = bops.contract(curr_r, curr_r, '0*', '0')
                    curr_projector = np.kron(curr_projector,
                                    projector_r.tensor.reshape([d**(2 * (N - 1- si)), d**(2 * (N - 1- si))]))
                projector += curr_projector / np.max(np.linalg.eigvalsh(curr_projector))
                rho_mps = bops.shiftWorkingSite(rho_mps, si, '<<')

            rho_mps = bops.vector_to_mps(rho_vec.reshape([d] * (2 * N)) \
                                         .transpose(np.array([[i, i + N] for i in range(N)]).reshape([2 * N])), d ** 2, N)
            for si in range(N - 2, 0, -1):
                if si > 0:
                    curr_l = tn.Node(np.eye(1))
                    for sli in range(si):
                        curr_l = bops.contract(curr_l, rho_mps[sli], [1 + sli], '0')
                    projector_l = bops.contract(curr_l, curr_l, [si + 1, '*'], [si + 1])
                    curr_projector = np.kron(projector_l.tensor.reshape([d ** (2 * si), d ** (2 * si)]),
                                             np.eye(d ** 2))
                else:
                    curr_projector = np.eye(d ** 2)
                if si < N - 1:
                    curr_r = tn.Node(np.eye(rho_mps[si + 1][0].dimension))
                    for sri in range(si + 1, N):
                        curr_r = bops.contract(curr_r, rho_mps[sri], [sri - si], '0')
                    projector_r = bops.contract(curr_r, curr_r, '0*', '0')
                    curr_projector = np.kron(curr_projector,
                                             projector_r.tensor.reshape(
                                                 [d ** (2 * (N - 1 - si)), d ** (2 * (N - 1 - si))]))
                projector -= curr_projector / np.max(np.linalg.eigvalsh(curr_projector))
                rho_mps = bops.shiftWorkingSite(rho_mps, si, '<<')
            projector = projector.reshape([d] * 4 * N).transpose([2 * i for i in range(N)]
                                                                 + [1 + 2 * i for i in range(N)]
                                                                 + [2 * N + 2 * i for i in range(N)]
                                                                 + [2 * N + 1 + 2 * i for i in range(N)])\
                .reshape([d**(2 * N), d**(2 * N)])
            rho_vec = np.matmul(evolver, rho_vec)

        if results_to == 'plot':
            plt.plot(J_expect)
        else:
            with open(outdir + '/explicit_J_expect', 'wb') as f:
                pickle.dump(J_expect, f)

    projectors_left, projectors_right = tdvp.get_initial_projectors(psi, L)
    I = np.eye(2).reshape([1, d**2, 1])
    J_expect = np.zeros(timesteps, dtype=complex)
    bond_dims = np.zeros(timesteps, dtype=complex)
    for ti in range(timesteps):
        print('---')
        print(ti)
        try:
            with open(newdir + '/mid_state_' + case + '_N_' + str(N)
                      + '_Omega_' + str(Omega) + '_nn_' + str(nn_num) + '_ti_' + str(ti), 'rb') as f:
                [ti, psi, projectors_left, projectors_right] = pickle.load(f)
            with open(newdir + '/tdvp_' + case + '_N_' + str(N)
                      + '_Omega_' + str(Omega) + '_nn_' + str(nn_num), 'rb') as f:
                [J_expect_form, bond_dims_form] = pickle.load(f)
                J_expect[:len(J_expect_form)] = J_expect_form
                bond_dims[:len(bond_dims_form)] = bond_dims_form
        except FileNotFoundError:
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
            bond_dims[ti] = psi[int(len(psi)/2)].tensor.shape[0]
            tdvp.tdvp_sweep(psi, L, projectors_left, projectors_right, dt, max_bond_dim=1024)
            if ti % save_each == 0:
                with open(newdir + '/tdvp_' + case + '_N_' + str(N) + '_Omega_' + str(Omega) + '_nn_' + str(nn_num), 'wb') as f:
                    pickle.dump([J_expect, bond_dims], f)
                with open(newdir + '/mid_state_' + case + '_N_' + str(N) + '_Omega_' + str(Omega) + '_nn_' + str(nn_num) + '_ti_' + str(ti), 'wb') as f:
                    pickle.dump([ti, psi, projectors_left, projectors_right], f)
    if results_to == 'plot':
        plt.plot(np.array(range(timesteps)) * 2, np.abs(J_expect))
        plt.show()