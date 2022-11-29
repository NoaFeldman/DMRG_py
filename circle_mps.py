import os.path
import basicOperations as bops
import numpy as np
import DMRG as dmrg
import tensornetwork as tn
import pickle
import sys
from os import path
import functools as ft
import scipy.linalg as linalg
import magicRenyi 

import string
digs = string.digits + string.ascii_letters
def int2base(x, base, str_len):
    if x < 0:
        sign = -1
    elif x == 0:
        return digs[0].zfill(str_len)
    else:
        sign = 1
    x *= sign
    digits = []
    while x:
        digits.append(digs[x % base])
        x = x // base
    if sign < 0:
        digits.append('-')

    digits.reverse()

    return ''.join(digits).zfill(str_len)


def get_H_terms(N, onsite_term, neighbor_term, d=2, bc='p'):
    if bc == 'p':
        onsite_terms = [np.kron(onsite_term, np.eye(d)) + np.kron(np.eye(d), onsite_term)
                        for i in range(int(N / 2))]
        onsite_terms[0] += neighbor_term
        neighbor_terms = [np.kron(neighbor_term.reshape([d] * 4), np.eye(d**2).reshape([d] * 4)).reshape([d**4, d**4])
                          + np.kron(np.eye(d**2).reshape([d] * 4), neighbor_term.reshape([d] * 4)).reshape([d**4, d**4])
                          for i in range(int(N / 2) - 1)]
        if N % 2 == 0:
            onsite_terms[-1] += neighbor_term
        else:
            onsite_terms = onsite_terms + [np.kron(onsite_term, np.eye(d))]
            neighbor_terms = neighbor_terms \
                             + [np.kron(neighbor_term.reshape([d] * 4), np.eye(d**2).reshape([d] * 4)).reshape([d**4, d**4]) + \
                                np.kron(neighbor_term.reshape([d] * 4), np.eye(d**2).reshape([d] * 4)).reshape([d] * 8)
                                    .transpose([1, 0, 2, 3, 5, 4, 6, 7]).reshape([d ** 4] * 2)]
        return onsite_terms, neighbor_terms
    else:
        onsite_terms = [onsite_term for i in range(N)]
        neighbor_terms = [neighbor_term for i in range(N - 1)]
        return onsite_terms, neighbor_terms


def minus_state(N):
    if N % 2 == 1:
        basic_node = np.array([1, -1, -1, 1], dtype=complex).reshape([1, 4, 1]) * 0.5
        right_node = np.array([1, 0, -1, 0], dtype=complex).reshape([1, 4, 1])
        state = [tn.Node(basic_node) for i in range(int(N/2))] + [tn.Node(right_node)]
        state[-1].tensor /= np.sqrt(bops.getOverlap(state, state))
        return state


def antiferromagnetic_state(N):
    if N % 2 == 1:
        basic_mid_site = np.zeros((4, 2, 4), dtype=complex)
        basic_mid_site[0, 0, 1] = 1
        basic_mid_site[1, 1, 0] = 1
        basic_mid_site[0, 0, 2] = 1
        basic_mid_site[2, 0, 3] = 1
        basic_mid_site[1, 1, 3] = 1
        basic_mid_site[3, 1, 2] = 1

        left_site = np.zeros((1, 2**2, 4**2))
        left_site[0, 1 + 0 * 2, 3 + 1 * 4] = 1
        left_site[0, 0 + 1 * 2, 2 + 0 * 4] = 1
        left_site[0, 1 + 0 * 2, 3 + 2 * 4] = 1
        left_site[0, 0 + 1 * 2, 2 + 3 * 4] = 1
        left_site[0, 0 + 0 * 2, 2 + 3 * 4] = 1
        left_site[0, 1 + 1 * 2, 3 + 2 * 4] = 1

        redundant_site = np.zeros((4, 2, 4))
        for di in range(4):
            redundant_site[di, 0, di] = 1
        down = np.copy(basic_mid_site)
        right_site = np.tensordot(down, redundant_site, [2, 0]).transpose([0, 3, 1, 2]).reshape([4**2, 2**2, 1])

        result = [tn.Node(left_site)]
        for i in range(1, int(N/2)):
            down = np.copy(basic_mid_site)
            up = np.copy(basic_mid_site)
            result.append(tn.Node(np.kron(down, up.transpose([2, 1, 0]))))
        result.append(tn.Node(right_site))
    else:
        basic_site = np.zeros((2, 2, 2), dtype=complex)
        basic_site[0, 0, 1] = 1
        basic_site[1, 1, 0] = 1
        result = [tn.Node(bops.contract(tn.Node(np.eye(4)),
                                tn.Node(np.kron(basic_site, basic_site).reshape([4, 4, 2**2, 4**2])),
                                '01', '01').tensor.reshape([1, 2**2, 4**2]))] + \
                 [tn.Node(np.kron(basic_site, basic_site)) for i in range(int(N/2) - 2)] + \
                 [tn.Node(bops.contract(tn.Node(np.kron(basic_site, basic_site).reshape([4**2, 2**2, 4, 4])),
                                tn.Node(np.eye(4)), '23', '01').tensor.reshape([4**2, 2**2, 1]))]
    for i in range(len(result) - 1):
        result[i], result[i+1], te = bops.svdTruncation(bops.contract(result[i], result[i+1], '2', '0'),
                                                        [0, 1], [2, 3], '>>')
    result[-1].tensor /= np.sqrt(bops.getOverlap(result, result))
    return result


def ferromagnetic_state(N):
    basic_mid_site = np.zeros((2, 2, 2), dtype=complex)
    basic_mid_site[0, 0, 0] = 1
    basic_mid_site[1, 1, 1] = 1

    redundant_site = np.zeros((2, 2, 2))
    for di in range(2):
        redundant_site[di, 0, di] = 1
    down = np.copy(basic_mid_site)
    right_site = np.tensordot(down, redundant_site, [2, 0]).transpose([0, 3, 1, 2]).reshape([2 ** 2, 2 ** 2, 1])
    left_site = np.tensordot(basic_mid_site, basic_mid_site, [2, 0]).transpose([1, 2, 0, 3]).reshape([1, 2 ** 2, 2**2])

    result = [tn.Node(left_site)]
    for i in range(1, int(N / 2)):
        down = np.copy(basic_mid_site)
        up = np.copy(basic_mid_site)
        result.append(tn.Node(np.kron(down, up.transpose([2, 1, 0]))))
    result.append(tn.Node(right_site))
    result[-1] /= np.sqrt(bops.getOverlap(result, result))
    return result


def memory_cheap_m2(psi, paulis, paulis_single=None):
    prev = tn.Node(np.eye(1).reshape([1] * 8))
    for si in range(len(psi) - 1):
        curr = np.zeros([psi[si][2].dimension]*8, dtype=complex)
        for pi in range(len(paulis)):
            for pj in range(len(paulis)):
                p_psi = bops.permute(bops.contract(
                    psi[si], tn.Node(np.kron(paulis[pi], paulis[pj])), '1', '0'), [0, 2, 1])
                curr += bops.contract(bops.contract(bops.contract(bops.contract(bops.contract(bops.contract(bops.contract(bops.contract(
                    prev, p_psi, '0', '0'), psi[si], '07', '01*'),
                    p_psi, '0', '0'), psi[si], '07', '01*'),
                    psi[si], '0', '0'), p_psi, '07', '01*'),
                    psi[si], '0', '0'), p_psi, '07', '01*').tensor

        prev = tn.Node(curr)
    prev = tn.Node(prev.tensor.transpose([0, 2, 4, 6, 1, 3, 5, 7]).reshape([prev[0].dimension**4]*2))
    if paulis_single is None:
        paulis_single = paulis
    single_site_magic_op = \
        np.kron(np.kron(paulis_single[0], paulis_single[0]), np.kron(paulis_single[0], paulis_single[0]).conj().T) + \
        np.kron(np.kron(paulis_single[1], paulis_single[1]), np.kron(paulis_single[1], paulis_single[1]).conj().T) + \
        np.kron(np.kron(paulis_single[2], paulis_single[2]), np.kron(paulis_single[2], paulis_single[2]).conj().T) + \
        np.kron(np.kron(paulis_single[3], paulis_single[3]), np.kron(paulis_single[3], paulis_single[3]).conj().T)
    single_site_magic_op = single_site_magic_op.reshape([d] * 8)
    return bops.contract(prev,
         bops.contract(bops.contract(
             tn.Node(np.kron(psi[-1].tensor, np.kron(psi[-1].tensor, np.kron(psi[-1].tensor, psi[-1].tensor)))),
             tn.Node(np.kron(single_site_magic_op, np.eye(d ** 4).reshape([d] * 8)).reshape([d ** 8, d ** 8])),
             '1', '0'),
             tn.Node(np.kron(psi[-1].tensor, np.kron(psi[-1].tensor, np.kron(psi[-1].tensor, psi[-1].tensor)))),
             '21', '12*'),
         '01', '01').tensor


def exact_H(N, ising_lambda):
    H = np.zeros((d**N, d**N))
    for i in range(N):
        H += np.kron(np.eye(d**i), np.kron(ising_lambda * X, np.eye(d**(N - i - 1))))
    for i in range(N-1):
        H += np.kron(np.eye(d ** i), np.kron(np.kron(Z, Z), np.eye(d ** (N - i - 2))))
    H += np.kron(Z, np.kron(np.eye(d**(N-2)), Z))
    vals, vecs = np.linalg.eigh(H)
    print(min(vals), vals[1])


def exact_m2(N, psi):
    vec = psi[0]
    for i in range(1, len(psi)):
        vec = bops.contract(vec, psi[i], [i+1], '0')
    vec = vec.tensor.reshape([d**(2*len(psi))])
    paulis = [np.eye(2), X, Y, Z]
    num_of_strings = len(paulis)**N
    result = 0
    for si in range(num_of_strings):
        string = int2base(si, len(paulis), N)
        mat = np.eye(d)
        for c in string:
            mat = np.kron(paulis[digs.index(c)], mat)
        result += np.abs(np.matmul(vec.conj().T, np.matmul(mat, vec)))**4
    return -np.log(result) / np.log(2) + N


d = 2
X = np.array([[0, 1], [1, 0]])
Z = np.diag([1, -1])
Y = np.array([[0, -1j], [1j, 0]])
unrotated_paulis = [np.eye(d), Z, Y, X]
angle_steps = 5 # 10
thetas = [np.pi * i / (2 * angle_steps) for i in range(angle_steps)]
phis = [np.pi * i / (2 * angle_steps) for i in range(angle_steps)]


def rotate_paulis(theta, phi):
    return [ft.reduce(np.matmul,
        [linalg.expm(1j * theta * Z), linalg.expm(1j * phi * X), unrotated_paulis[i], linalg.expm(-1j * phi * X), linalg.expm(-1j * theta * Z)])
            for i in range(len(unrotated_paulis))]


def filename(dirname, J, N, ising_lambda, boundary_conditions, bc):
    file_name = dirname + '/ising_' + boundary_conditions + '_J_' + str(J) + '_N_' + str(N) + '_lambda_' + str(ising_lambda)
    if bc == 'p':
        return file_name
    else:
        return file_name + '_' + bc



def imps_ground_state(N, J, ising_lambda):
    h_term = ising_lambda * np.kron(X, np.eye(2)) + J * np.kron(Z, Z)
    h_term = tn.Node(h_term.reshape([2] * 4))
    site = tn.Node(np.sqrt(1/2) * np.array([1, 1]).reshape(1, 2, 1))
    pair = bops.contract(site, site, '2', '0')
    steps = 200
    for i in range(steps):
        [r, s, l, te] = bops.svdTruncation(bops.permute(bops.contract(pair, h_term, '12', '01'), [0, 2, 3, 1]),
                                           [0, 1], [2, 3], '>*<', maxBondDim=32)
        pair = bops.contract(bops.contract(s, l, '1', '0'), r, '2', '0')
    [r, s, l, te] = bops.svdTruncation(pair, [0, 1], [2, 3], '>*<')
    site = r
    gs = [tn.Node(np.tensordot(site.tensor, site.tensor, [[0], [2]]).transpose([0, 3, 1, 2]).reshape([1, 2**2, site.tensor.shape[0]*site.tensor.shape[2]]))] + \
         [tn.Node(np.kron(site.tensor, site.tensor.transpose([2, 1, 0])))] * int((N - 3) / 2) + \
         [tn.Node(np.kron(site.tensor, np.array([1, 0]).reshape([1, 2, 1])).transpose([0, 2, 1]).reshape([site.tensor.shape[0]*site.tensor.shape[2], 2**2, 1]))]
    for i in range(len(gs) - 1):
        pair = bops.contract(gs[i], gs[i+1], '2', '0')
        gs[i], gs[i+1], te = bops.svdTruncation(pair, [0, 1], [2, 3], '>>', maxBondDim=4)
    gs[-1].tensor /= np.sqrt(bops.getOverlap(gs, gs))
    return gs


def ground_states_magic(N, J, ising_lambdas, dirname, bc='p'):
    all_m2s_0_basis = np.zeros(len(ising_lambdas), dtype=complex)
    all_m2s_min_basis = np.zeros(len(ising_lambdas), dtype=complex)
    all_alphas_squared = np.zeros(len(ising_lambdas))
    all_zz = np.zeros(len(ising_lambdas))

    if J == -1:
        if bc == 'p':
            psi_0 = ferromagnetic_state(N)
        else:
            psi_0 = bops.getStartupState(N, d=2)
    else:
        if bc == 'p':
            psi_0 = antiferromagnetic_state(N)
        else:
            psi_0 = bops.getStartupState(N, d=2)
    for li in range(len(ising_lambdas)):
        ising_lambda = ising_lambdas[li]
        results_filename = filename(dirname, J, N, ising_lambda, 'PBC', bc=bc)
        if os.path.exists(results_filename):
            data = pickle.load(open(results_filename, 'rb'))
        else:
            onsite_terms, neighbor_terms = get_H_terms(N, ising_lambda * X, J * np.kron(Z, Z), bc=bc)
            if bc == 'p':
                d = 4
            else:
                d = 2
            gs, E0, trunc_errs = dmrg.DMRG(psi_0, onsite_terms, neighbor_terms, d=d, initial_bond_dim=16, maxBondDim=512,
                                           accuracy=1e-10)

            # TODO allow obc
            if bc == 'p':
                x_string = [tn.Node(np.kron(X, X))] * (len(gs) - 1) + [tn.Node(np.kron(X, np.eye(2)))]
                left_op = tn.Node(-1 * np.kron(X, np.eye(2)))
            else:
                x_string = [tn.Node(X)] * len(gs)
                left_op = tn.Node(X)
            if np.abs(np.round(bops.getExpectationValue(gs, x_string), 4)) != 1:
                gsx = bops.copyState(gs)
                for i in range(len(gsx) - 1):
                    gsx[i] = bops.permute(bops.contract(gsx[i], x_string[i], '1', '0'), [0, 2, 1])
                gsx[-1] = bops.permute(bops.contract(gsx[-1], left_op, '1', '0'), [0, 2, 1])
                gs = bops.addStates(gs, gsx)
                gs[-1] /= np.sqrt(bops.getOverlap(gs, gs))


            # split sites so it is consistent with magicRenyi.getRenyiEntropy
            relaxed = bops.relaxState(gs, 4)
            state_accuracy = bops.getOverlap(gs, relaxed)
            m2s = np.zeros((angle_steps, angle_steps), dtype=complex)
            for ti in range(angle_steps):
                for pi in range(angle_steps):
                    paulis = rotate_paulis(thetas[ti], phis[pi])
                    if bc == 'p':
                        m2 = memory_cheap_m2(relaxed, paulis)
                    else:
                        m2 =  magicRenyi.getSecondRenyi_basis(relaxed, 2, thetas[ti], phis[pi], 0)
                    m2s[ti, pi] = m2
            if bc == 'p':
                single_site_rdm = bops.contract(gs[0], gs[0], '01', '01*')
                for i in range(1, len(gs) - 1):
                    single_site_rdm = bops.contract(bops.contract(
                        single_site_rdm, gs[i], '0', '0'), gs[i], '01', '01*')

                # TODO maybe alpha_squared should be redone as in the if False part (non-canonical)
                single_site_rdm = bops.contract(tn.Node(bops.contract(gs[-1], gs[-1], '02', '02*').tensor.reshape([d] * 4)),
                                            tn.Node(np.eye(d)), '13', '01').tensor
            else:
                single_site_rdm = bops.contract(gs[-1], gs[-1], '02', '02*').tensor
            alpha_squared = sum([np.matmul(single_site_rdm, P).trace()**2 for P in [X, Y, Z]])
            psi_0 = gs
            data = [gs, state_accuracy, m2s, alpha_squared]
        if len(data) == 4 and different_bases:
            gs = data[0]
            relaxed = bops.relaxState(gs, 4)
            m2s_singled_out_site = np.zeros((angle_steps, angle_steps, angle_steps, angle_steps), dtype=complex)
            for ti in range(angle_steps):
                for pi in range(angle_steps):
                    paulis = rotate_paulis(thetas[ti], phis[pi])
                    for ti_l in range(angle_steps):
                        for pi_l in range(angle_steps):
                            paulis_left = rotate_paulis(thetas[ti_l], phis[pi_l])
                            if bc == 'p':
                                m2 = memory_cheap_m2(relaxed, paulis, paulis_left)
                            else:
                                m2 =  magicRenyi.getSecondRenyi_basis(relaxed, 2, thetas[ti], phis[pi], 0)
                            m2s_singled_out_site[ti, pi, ti_l, pi_l] = m2
            data.append(m2s_singled_out_site)
            pickle.dump(data, open(results_filename, 'wb'))
        [gs, state_accuracy, m2s, alpha_squared] = data[:4]
        if different_bases:
            m2s_singled_out_site = data[4]
        print(J, N, ising_lambda)
        if bc == 'p':
            if different_bases:
                all_m2s_0_basis[li] = -(np.log(m2s_singled_out_site[0, 0, 0, 0])/ np.log(2) - N)
                all_m2s_min_basis[li] = np.amin(-(np.log(m2s_singled_out_site)/ np.log(2) - N))
            else:
                all_m2s_0_basis[li] = -(np.log(m2s[0, 0])/ np.log(2) - N)
                all_m2s_min_basis[li] = np.amin(-(np.log(m2s)/ np.log(2) - N))
        else:
            all_m2s_0_basis[li] = m2s[0, 0]
            all_m2s_min_basis[li] = np.amin(m2s)
        all_alphas_squared[li] = alpha_squared
    return all_m2s_0_basis, all_m2s_min_basis, all_alphas_squared, all_zz


dirname = sys.argv[1]
Ns = [i * 2 + 1 for i in range(int(sys.argv[2]), int(sys.argv[3]))]
different_bases = True
lambda_step = 0.1
lambda_critical_step = 0.01
phase_transition = 1
ising_lambdas = [np.round(lambda_step * i, 8) for i in range(1, int(phase_transition / lambda_step))] \
    + [np.round(phase_transition + lambda_critical_step * i, 8) for i in range(-9, 10)] \
    + [np.round(lambda_step * i, 8) for i in range(int((phase_transition + lambda_step) / lambda_step), int(2.5 / lambda_step))]
bc = 'p'


m2s = np.zeros((len(Ns), len(ising_lambdas)), dtype=complex)
m2s_min = np.zeros((len(Ns), len(ising_lambdas)), dtype=complex)
alphas_squared = np.zeros((len(Ns), len(ising_lambdas)))
m2s_ferro = np.zeros((len(Ns), len(ising_lambdas)), dtype=complex)
m2s_min_ferro = np.zeros((len(Ns), len(ising_lambdas)), dtype=complex)
alphas_squared_ferro = np.zeros((len(Ns), len(ising_lambdas)))
zz = np.zeros((len(Ns), len(ising_lambdas)))
zz_ferro = np.zeros((len(Ns), len(ising_lambdas)))
for Ni in range(len(Ns)):
    N = Ns[Ni]
    curr_m2s_0_basis_ferro, curr_m2s_min_basis_ferro, curr_alphas_squared_ferro, curr_zz_ferro = \
        ground_states_magic(N, -1, ising_lambdas, dirname, bc)
    curr_m2s_0_basis, curr_m2s_min_basis, curr_alphas_squared, curr_zz = ground_states_magic(N, 1, ising_lambdas, dirname, bc)
    m2s[Ni, :] = curr_m2s_0_basis
    m2s_min[Ni, :] = curr_m2s_min_basis
    alphas_squared[Ni, :] = curr_alphas_squared
    zz[Ni, :] = curr_zz
    m2s_ferro[Ni, :] = curr_m2s_0_basis_ferro
    m2s_min_ferro[Ni, :] = curr_m2s_min_basis_ferro
    alphas_squared_ferro[Ni, :] = curr_alphas_squared_ferro
    zz_ferro[Ni, :] = curr_zz_ferro

def full_plot():
    import matplotlib.pyplot as plt
    ff, axs = plt.subplots(2, 2)
    for li in range(len(ising_lambdas)):
        axs[0, 0].plot(Ns, m2s[:, li])
        axs[0, 1].plot(Ns, m2s_min[:, li])
        axs[1, 0].plot(Ns, m2s_ferro[:, li])
        axs[1, 1].plot(Ns, m2s_min_ferro[:, li])
    axs[1, 0].legend([str(ising_lambdas[li]) for li in range(len(ising_lambdas))])
    plt.show()
    ff, axs = plt.subplots(2, 3)
    m = axs[0, 0].pcolormesh(ising_lambdas, Ns, np.real(m2s), shading='auto')
    plt.colorbar(m, ax=axs[0, 0])
    axs[0, 0].set_title(r'$m_2$')
    m = axs[0, 1].pcolormesh(ising_lambdas, Ns, np.real(m2s_min), shading='auto')
    plt.colorbar(m, ax=axs[0, 1])
    axs[0, 1].set_title(r'min$(m_2)$')
    m = axs[0, 2].pcolormesh(ising_lambdas, Ns, np.real(alphas_squared), shading='auto')
    plt.colorbar(m, ax=axs[0, 2])
    axs[0, 2].set_title(r'$|\alpha|^2$')
    m = axs[1, 0].pcolormesh(ising_lambdas, Ns, np.real(m2s_ferro), shading='auto')
    plt.colorbar(m, ax=axs[1, 0])
    m = axs[1, 1].pcolormesh(ising_lambdas, Ns, np.real(m2s_min_ferro), shading='auto')
    plt.colorbar(m, ax=axs[1, 1])
    m = axs[1, 2].pcolormesh(ising_lambdas, Ns, np.real(alphas_squared_ferro), shading='auto')
    plt.colorbar(m, ax=axs[1, 2])
    for j in range(3):
        axs[1, j].set_xlabel(r'$\lambda$')
    axs[0, 0].set_ylabel('antiferromagnetic \n N')
    axs[1, 0].set_ylabel('ferromagnetic \n N')
    plt.show()


def check_log_dependence():
    import matplotlib.pyplot as plt
    J = -1
    lambda_inds = [5, 6, 7, 8, 9, 10]
    for li in lambda_inds:
        plt.plot(Ns, np.exp(m2s_min_ferro[:, li]))
    plt.legend([str(ising_lambdas[li]) for li in lambda_inds])
    plt.show()
full_plot()
# check_log_dependence()

