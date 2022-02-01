import basicOperations as bops
import numpy as np
import DMRG as dmrg
import tensornetwork as tn
import pickle
import basicDefs as basic
import matplotlib.pyplot as plt
import magicRenyi
import sys
import random

d = 2
model = sys.argv[1]
indir = sys.argv[2]
if len(sys.argv) == 5:
    n = 16
    curr_ind = 3
else:
    n = int(sys.argv[3])
    curr_ind = 4
range_i = int(sys.argv[curr_ind])
range_f = int(sys.argv[curr_ind + 1])

def get_xxz_dmrg_terms(delta):
    onsite_terms = [0 * np.eye(d) for i in range(n)]
    neighbor_terms = [np.kron(basic.pauli2Z, basic.pauli2Z) * delta + \
                      np.kron(basic.pauli2X, basic.pauli2X) + \
                      np.kron(basic.pauli2Y, basic.pauli2Y) for i in range(n - 1)]
    return onsite_terms, neighbor_terms


def get_t_ising_dmrg_terms(h):
    onsite_terms = [- h * basic.pauli2X for i in range(n)]
    neighbor_terms = [- np.kron(basic.pauli2Z, basic.pauli2Z) for i in range(n - 1)]
    return onsite_terms, neighbor_terms


def get_xy_dmrg_terms(gamma):
    onsite_terms = [0 * np.eye(2) for i in range(n)]
    neighbor_terms = [-(1 + gamma) * np.kron(basic.pauli2X, basic.pauli2X) -
                      (1 - gamma) * np.kron(basic.pauli2Y, basic.pauli2Y) for i in range(n - 1)]
    return onsite_terms, neighbor_terms


t_z = np.array([[0, np.exp(1j * np.pi / 4)], [np.exp(-1j * np.pi / 4), 0]])
hadamard = np.array([[1, 1, ], [1, -1]]) / np.sqrt(2)
rotated_t_gate = np.matmul(hadamard, np.matmul(t_z, hadamard))
def get_magic_ising_dmrg_terms(h):
    onsite_terms = [- h * t_z for i in range(n)]
    neighbor_terms = [- np.kron(basic.pauli2Z, basic.pauli2Z) for i in range(n - 1)]
    return onsite_terms, neighbor_terms

def get_magic_xxz_dmrg_terms(delta):
    onsite_terms = [0 * np.eye(d) for i in range(n)]
    neighbor_terms = [np.kron(t_z, t_z) * delta + \
                      np.kron(basic.pauli2X, basic.pauli2X) + \
                      np.kron(basic.pauli2Y, basic.pauli2Y) for i in range(n - 1)]
    return onsite_terms, neighbor_terms

def get_magic_xxz_real_pairs_dmrg_terms(delta):
    onsite_terms = [0 * np.eye(d) for i in range(n)]
    neighbor_terms = [np.kron(t_z, np.conj(t_z)) * delta + \
                      np.kron(basic.pauli2X, basic.pauli2X) + \
                      np.kron(basic.pauli2Y, basic.pauli2Y) for i in range(n - 1)]
    return onsite_terms, neighbor_terms

def get_xy_magic_dmrg_terms(gamma):
    onsite_terms = [0 * np.eye(2) for i in range(n)]
    neighbor_terms = [-(1 + gamma) * np.kron(basic.pauli2X, basic.pauli2X) -
                      (1 - gamma) * np.kron(t_z, t_z) for i in range(n - 1)]
    return onsite_terms, neighbor_terms


def filename(indir, model, param_name, param, n):
    if n == 16:
        return indir + '/magic/results/' + model + '/' + param_name + '_' + str(param)
    else:
        return indir + '/magic/results/' + model + '/' + param_name + '_' + str(param) + '_n_' + str(n)


def get_H_explicit(onsite_terms, neighbor_terms):
    H = np.zeros((d**n, d**n), dtype='complex')
    for i in range(n):
        curr = np.eye(1)
        for j in range(i):
            curr = np.kron(curr, np.eye(2))
        curr = np.kron(curr, onsite_terms[i])
        for j in range(i + 1, n):
            curr = np.kron(curr, np.eye(2))
        H += curr
    for i in range(n-1):
        curr = np.eye(1)
        for j in range(i):
            curr = np.kron(curr, np.eye(2))
        curr = np.kron(curr, neighbor_terms[i])
        for j in range(i + 2, n):
            curr = np.kron(curr, np.eye(2))
        H += curr
    return H

def get_half_system_dm(psi):
    curr = bops.contract(psi[int(n / 2)], psi[int(n / 2)], '0', '0*')
    for site in range(int(n / 2) + 1, n):
        curr = bops.contract(curr, psi[site], [2 * (site - int(n / 2)) - 1], '0')
        curr = bops.contract(curr, psi[site], [2 * (site - int(n / 2))], '0*')
    reorder = [2 * j for j in range(int(n / 2))] + [2 * j + 1 for j in range(int(n / 2) - 1)] + [n, n - 1,
                                                                                                 n + 1]  # [0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 16, 15, 17]
    dm = curr.tensor.transpose(reorder).reshape([d ** int(n / 2), d ** int(n / 2)])
    return dm


if model == 'xxz':
    param_name = 'delta'
    params = [np.round(i * 0.1 - 2, 1) for i in range(range_i, range_f)]
    h_func = get_xxz_dmrg_terms
elif model == 'magic_xxz':
    param_name = 'delta'
    params = [np.round(i * 0.1 - 2, 1) for i in range(range_i, range_f)]
    h_func = get_magic_xxz_dmrg_terms
elif model == 't_ising':
    param_name = 'h'
    params = [np.round(i * 0.1, 1) for i in range(range_i, range_f)]
    h_func = get_t_ising_dmrg_terms
elif model == 'ising_magic':
    param_name = 'h'
    params = [np.round(i * 0.1, 1) for i in range(20)]
    h_func = get_magic_ising_dmrg_terms
elif model == 'xy':
    param_name = 'gamma'
    params = [np.round(i * 0.1, 1) for i in range(-10, 11)]
    h_func = get_xy_dmrg_terms
elif model == 'xy_magic':
    param_name = 'gamma'
    params = [np.round(i * 0.1, 1) for i in range(-10, 10)]
    h_func = get_xy_magic_dmrg_terms

def run():
    psi = bops.getStartupState(n, d)
    p2s = np.zeros(len(params))
    thetas = [0.0, 0.15, 0.25, 0.4]
    phis = [0.0, 0.15, 0.25, 0.4]
    etas = [0.0, 0.15, 0.25, 0.4]
    m2s = np.zeros((len(params), len(thetas), len(phis), len(etas)))
    mhalves = np.zeros((len(params), len(thetas), len(phis), len(etas)))
    best_bases = []
    worst_bases = []
    szs = np.zeros(len(params), dtype=complex)
    for pi in range(len(params)):
        param = params[pi]
        try:
            with open(filename(indir, model, param_name, param, n), 'rb') as f:
                [psi, m2, m2_optimized, best_basis, mhalf, m2_maximized, worst_basis,
                 mhalf_optimized, mhalf_best_basis, mhalf_maximized, mhalf_worst_basis] = pickle.load(f)
        except FileNotFoundError or EOFError:
            onsite_terms, neighbor_terms = h_func(param)
            psi_bond_dim = psi[int(n/2)].tensor.shape[0]
            psi, E0, truncErrs = dmrg.DMRG(psi, onsite_terms, neighbor_terms, accuracy=1e-12)
            if n == 12:
                H = get_H_explicit(onsite_terms, neighbor_terms)
                vals = np.linalg.eigvalsh(H)
                print([E0, vals[0]])
            if psi[int(n/2)].tensor.shape[0] > 4:
                psi_copy = bops.relaxState(psi, 4)
                print(bops.getOverlap(psi, psi_copy))
                m2 = magicRenyi.getSecondRenyi(psi_copy, d)
                m2_optimized, best_basis = magicRenyi.getSecondRenyi_optimizedBasis(psi_copy, d)
                m2_maximized, worst_basis = magicRenyi.getSecondRenyi_optimizedBasis(psi_copy, d, opt='max')
            else:
                m2 = magicRenyi.getSecondRenyi(psi, d)
                m2_optimized, best_basis = magicRenyi.getSecondRenyi_optimizedBasis(psi, d)
                m2_maximized, worst_basis = magicRenyi.getSecondRenyi_optimizedBasis(psi, d, opt='max')
            dm = get_half_system_dm(psi)
            mhalf = magicRenyi.getHalfRenyiExact_dm(dm, d)
            mhalf_optimized, mhalf_best_basis, mhalf_maximized, mhalf_worst_basis = \
                magicRenyi.getHalfRenyiExact_dm_optimized(dm, d)
            szs[pi] = sum([bops.getExpectationValue(psi, [tn.Node(np.eye(2)) for i in range(j)] + \
                                                    [tn.Node(basic.pauli2Z)] + \
                                                    [tn.Node(np.eye(2)) for i in range(n - 1 - j)]) for j in range(n)])
        print(psi[int(n/2)].tensor.shape)
        if psi[int(n / 2)].tensor.shape[0] > 4:
            psi_copy = bops.relaxState(psi, 4)
            print(bops.getOverlap(psi, psi_copy))
        dm = get_half_system_dm(psi)
        for ti in range(len(thetas)):
            theta = thetas[ti]
            for phi_i in range(len(phis)):
                phi = phis[phi_i]
                for ei in range(len(etas)):
                    eta = etas[ei]
                    m2s[pi, ti, phi_i, ei] = magicRenyi.getSecondRenyi_basis(psi, d, theta, phi, eta)
                    mhalves[pi, ti, phi_i, ei] = magicRenyi.getHalfRenyiExact_dm_basis(dm, d, theta, phi, eta)
        p2s[pi] = bops.getRenyiEntropy(psi, 2, int(n / 2))
        best_bases.append([phase / np.pi for phase in best_basis])
        worst_bases.append([phase / np.pi for phase in worst_basis])
        print(param)
        with open(filename(indir, model, param_name, param, n), 'wb') as f:
            pickle.dump([psi, m2, m2_optimized, best_basis, mhalf, m2_maximized, worst_basis,
                         mhalf_optimized, mhalf_best_basis, mhalf_maximized, mhalf_worst_basis], f)
    with open(indir + '/magic/results/m2s_' + param_name + 's_' + str(params[range_i]) + '_' + str(params[range_f - 1]), 'wb') as f:
        pickle.dump(m2s, f)
    with open(indir + '/magic/results/mhalves_' + param_name + 's_' + str(params[range_i]) + '_' + str(params[range_f - 1]), 'wb') as f:
        pickle.dump(mhalves, f)
    # f, axs = plt.subplots(3, 1, gridspec_kw={'wspace':0, 'hspace':0}, sharex='all')
    # for ti in range(len(thetas)):
    #     for phi_i in range(len(phis)):
    #         for ei in range(len(etas)):
    #             axs[1].plot(params, m2s[:, ti, phi_i, ei])
    # for ti in range(len(thetas)):
    #     for phi_i in range(len(phis)):
    #         for ei in range(len(etas)):
    #             axs[2].plot(params, mhalves[:, ti, phi_i, ei])
    # axs[0].plot(params, p2s)
    # axs[0].plot(params, np.real(szs) / max(1, np.max(np.abs(szs))))
    # axs[0].legend([r'$p_2$', r'$S_z / $' + str(np.round(max(1, np.max(np.abs(szs))), 1))])
    # plt.title(model)
    # plt.xlabel(param_name)
    # plt.show()

run()

def analyze_scaling():
    ns = [8, 12, 16, 20]
    Szs = np.zeros((len(ns), len(params)))
    p2s = np.zeros((len(ns), len(params)))
    m2s = np.zeros((len(ns), len(params)))
    m2s_maximized = np.zeros((len(ns), len(params)))
    m2s_optimized = np.zeros((len(ns), len(params)))
    mhalves = np.zeros((len(ns), len(params)))
    mhalves_maximized = np.zeros((len(ns), len(params)))
    mhalves_optimized = np.zeros((len(ns), len(params)))
    f, axs = plt.subplots(2, gridspec_kw={'wspace': 0, 'hspace': 0}, sharex='all')
    for pi in [4 * j for j in range(int(len(params) / 4))]:
        param = params[pi]
        for ni in range(len(ns)):
            n = ns[ni]
            with open(filename(indir, model, param_name, param, n), 'rb') as f:
                [psi, m2, m2_optimized, best_basis, mhalf, m2_maximized, worst_basis,
                 mhalf_optimized, mhalf_best_basis, mhalf_maximized, mhalf_worst_basis] = pickle.load(f)
            Szs[ni, pi] = sum([bops.getExpectationValue(psi, [tn.Node(np.eye(2)) for i in range(j)] + \
                            [tn.Node(basic.pauli2Z)] + [tn.Node(np.eye(2)) for i in range(n - 1 - j)])
                               for j in range(n)])
            p2s[ni, pi] = bops.getRenyiEntropy(psi, 2, int(n/2))
            m2s[ni, pi] = m2
            m2s_optimized[ni, pi] = m2_optimized
            m2s_maximized[ni, pi] = m2_maximized
            mhalves[ni, pi] = mhalf
            mhalves_maximized[ni, pi] = mhalf_maximized
            mhalves_optimized[ni, pi] = mhalf_optimized
        color = ["#"+''.join([random.choice('ABCDEF0123456789') for i in range(6)])]
        axs[0].plot(ns, p2s[:, pi], color=color[0])
        axs[1].plot(ns, m2s[:, pi], color=color[0])
        axs[1].plot(ns, m2s_optimized[:, pi], '--k', color=color[0])
        axs[1].plot(ns, m2s_maximized[:, pi], ':k', color=color[0])
    plt.show()

# analyze_scaling()