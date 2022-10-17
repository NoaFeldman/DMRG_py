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


import string
digs = string.digits + string.ascii_letters
def int2base(x, base):
    if x < 0:
        sign = -1
    elif x == 0:
        return digs[0]
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

    return ''.join(digits)


def get_H_terms(N, onsite_term, neighbor_term, d=2):
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


def memory_cheap_m2(psi, paulis):
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
    single_site_magic_op = \
        np.kron(np.kron(paulis[0], paulis[0]), np.kron(paulis[0], paulis[0]).conj().T) + \
        np.kron(np.kron(paulis[1], paulis[1]), np.kron(paulis[1], paulis[1]).conj().T) + \
        np.kron(np.kron(paulis[2], paulis[2]), np.kron(paulis[2], paulis[2]).conj().T) + \
        np.kron(np.kron(paulis[3], paulis[3]), np.kron(paulis[3], paulis[3]).conj().T)
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
    vec = vec.tensor.reshape([4**len(psi)])
    num_of_strings = len(paulis)**N
    result = 0
    for st in range(num_of_strings):
        op = ft.reduce(np.kron, [paulis[int(s, 2)] for s in
                list(map(''.join, zip(*[iter(bin(st).split('b')[1].zfill(2 * N))]*2)))] + [np.eye(d)])
        # op = ft.reduce(np.kron, [paulis[int(s, 2)] for s in list(map(''.join, zip(*[iter(bin(st).split('b')[1].zfill(N))]*1)))])
        result += np.abs(np.matmul(vec.conj().T, np.matmul(op, vec)))**4
    return result


d = 2
X = np.array([[0, 1], [1, 0]])
Z = np.diag([1, -1])
Y = np.array([[0, -1j], [1j, 0]])
unrotated_paulis = [np.eye(d), Z, Y, X]
angle_steps = 10
thetas = [np.pi * i / (2 * angle_steps) for i in range(angle_steps)]
phis = [np.pi * i / (2 * angle_steps) for i in range(angle_steps)]


def rotate_paulis(theta, phi):
    return [ft.reduce(np.matmul,
        [linalg.expm(1j * theta * Z), linalg.expm(1j * phi * X), unrotated_paulis[i], linalg.expm(-1j * phi * X), linalg.expm(-1j * theta * Z)])
            for i in range(len(unrotated_paulis))]


# for ti in range(angle_steps):
#     for pi in range(angle_steps):
#         theta = thetas[ti]
#         phi = phis[pi]
#         paulis = rotate_paulis(theta, phi)
#         is_ids = [np.round(np.matmul(p, p), 14) for p in paulis]
#         is_anticommuting = [np.round(np.matmul(p, q) + np.matmul(q, p), 14) for q in paulis[1:] for p in paulis[1:]]
#         is_algebra = [[[np.round(np.matmul(q, p) - 1j * r, 14) for q in paulis[1:]] for p in paulis[1:]] for r in paulis[1:]]
#         cyclic = [is_algebra[0][1][2], is_algebra[1][2][0], is_algebra[2][0][1]]
#         is_algebra = [[[np.round(np.matmul(q, p) + 1j * r, 14) for q in paulis[1:]] for p in paulis[1:]] for r in paulis[1:]]
#         anti_cyclic = [is_algebra[0][2][1], is_algebra[1][0][2], is_algebra[2][1][0]]
#         dbg = 1


def filename(dirname, J, N, ising_lambda, boundary_conditions):
    return dirname + '/ising_' + boundary_conditions + '_J_' + str(J) + '_N_' + str(N) + '_lambda_' + str(ising_lambda)


def add_mps(psi1, psi2):
    result = []
    for i in range(len(psi1)):
        ten = np.zeros((psi1[i][0].dimension + psi2[i][0].dimension, d**2, psi1[i][2].dimension + psi2[i][2].dimension), dtype=complex)
        ten[:psi1[i][0].dimension, :, :psi1[i][2].dimension] = psi1[i].tensor
        ten[psi1[i][0].dimension:, :, psi1[i][2].dimension:] = psi2[i].tensor
        result.append(tn.Node(ten))
    result[0] = bops.contract(tn.Node(np.ones((1, 2))), result[0], '1', '0')
    result[-1] = bops.contract(result[-1], tn.Node(np.ones((2, 1))), '2', '0')
    return result


def ground_states_magic(N, J, ising_lambdas):
    all_m2s_0_basis = np.zeros(len(ising_lambdas), dtype=complex)
    all_m2s_min_basis = np.zeros(len(ising_lambdas), dtype=complex)
    all_alphas_squared = np.zeros(len(ising_lambdas))

    psi_0 = minus_state(N)
    for li in range(len(ising_lambdas)):
        ising_lambda = ising_lambdas[li]
        print(N, ising_lambda)
        results_filename = filename('magic/results/circle_ising', J, N, ising_lambda, 'PBC')
        if os.path.exists(results_filename):
            [gs, state_accuracy, m2s, alpha_squared] = pickle.load(open(results_filename, 'rb'))
        else:
            onsite_terms, neighbor_terms = get_H_terms(N, ising_lambda * X, J * np.kron(Z, Z))

            gs, E0, trunc_errs = dmrg.DMRG(psi_0, onsite_terms, neighbor_terms, d=4, initial_bond_dim=16, maxBondDim=512,
                                           accuracy=1e-10)
            # split sites so it is consistent with magicRenyi.getRenyiEntropy
            relaxed = bops.relaxState(gs, 4)
            state_accuracy = bops.getOverlap(gs, relaxed)
            m2s = np.zeros((angle_steps, angle_steps), dtype=complex)
            for ti in range(angle_steps):
                for pi in range(angle_steps):
                    paulis = rotate_paulis(thetas[ti], phis[pi])
                    m2 = memory_cheap_m2(relaxed, paulis)
                    m2s[ti, pi] = m2
            single_site_rdm = bops.contract(tn.Node(bops.contract(gs[-1], gs[-1], '02', '02*').tensor.reshape([d] * 4)),
                                            tn.Node(np.eye(d)), '13', '01').tensor
            print(single_site_rdm)
            alpha_squared = sum([np.matmul(single_site_rdm, P).trace()**2 for P in [X, Y, Z]])
            psi_0 = gs
            pickle.dump([gs, state_accuracy, m2s, alpha_squared], open(results_filename, 'wb'))
        print(state_accuracy)
        all_m2s_0_basis[li] = (np.log(m2s[0, 0]) - N)/ np.log(2)
        all_m2s_min_basis[li] = (np.log(np.amin(m2s)) - N)/ np.log(2)
        all_alphas_squared[li] = alpha_squared
    return all_m2s_0_basis, all_m2s_min_basis, all_alphas_squared


Ns = [i * 2 + 1 for i in range(3, 10)]
lambda_step = 0.1
ising_lambdas = [np.round(lambda_step * i, 8) for i in range(-int(3 / lambda_step), int(3 / lambda_step))]

m2s = np.zeros((len(Ns), len(ising_lambdas)), dtype=complex)
m2s_min = np.zeros((len(Ns), len(ising_lambdas)), dtype=complex)
alphas_squared = np.zeros((len(Ns), len(ising_lambdas)))
m2s_ferro = np.zeros((len(Ns), len(ising_lambdas)), dtype=complex)
m2s_min_ferro = np.zeros((len(Ns), len(ising_lambdas)), dtype=complex)
alphas_squared_ferro = np.zeros((len(Ns), len(ising_lambdas)))
for Ni in range(len(Ns)):
    N = Ns[Ni]
    curr_m2s_0_basis, curr_m2s_min_basis, curr_alphas_squared = ground_states_magic(N, 1, ising_lambdas)
    curr_m2s_0_basis_ferro, curr_m2s_min_basis_ferro, curr_alphas_squared_ferro = ground_states_magic(N, -1, ising_lambdas)
    m2s[Ni, :] = curr_m2s_0_basis
    m2s_min[Ni, :] = curr_m2s_min_basis
    alphas_squared[Ni, :] = curr_alphas_squared
    m2s_ferro[Ni, :] = curr_m2s_0_basis_ferro
    m2s_min_ferro[Ni, :] = curr_m2s_min_basis_ferro
    alphas_squared_ferro[Ni, :] = curr_alphas_squared_ferro

plot = False
if plot:
    import matplotlib.pyplot as plt
    ff, axs = plt.subplots(2, 3)
    m = axs[0, 0].pcolormesh(ising_lambdas, Ns, np.real(m2s))
    plt.colorbar(m, ax=axs[0, 0])
    m = axs[0, 1].pcolormesh(ising_lambdas, Ns, np.real(m2s_min))
    plt.colorbar(m, ax=axs[0, 1])
    m = axs[0, 2].pcolormesh(ising_lambdas, Ns, np.real(alphas_squared))
    plt.colorbar(m, ax=axs[0, 2])
    m = axs[1, 0].pcolormesh(ising_lambdas, Ns, np.real(m2s_ferro))
    plt.colorbar(m, ax=axs[1, 0])
    m = axs[1, 1].pcolormesh(ising_lambdas, Ns, np.real(m2s_min_ferro))
    plt.colorbar(m, ax=axs[1, 1])
    m = axs[1, 2].pcolormesh(ising_lambdas, Ns, np.real(alphas_squared_ferro))
    plt.colorbar(m, ax=axs[1, 2])
    for i in range(2):
        for j in range(3):
            axs[i, j].set_xlabel(r'$\lambda$')
            axs[i, j].set_ylabel('N')
    plt.show()