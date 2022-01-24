import basicOperations as bops
import numpy as np
import DMRG as dmrg
import tensornetwork as tn
import pickle
import basicDefs as basic
import matplotlib.pyplot as plt
import magicRenyi
import sys
import os

d = 2
model = sys.argv[1]
indir = sys.argv[2]
if len(sys.argv) == 3:
    n = 16
else:
    n = int(sys.argv[3])


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

if model == 'xxz':
    param_name = 'delta'
    params = [np.round(i * 0.1 - 2, 1) for i in range(1, 40)]
    h_func = get_xxz_dmrg_terms
elif model == 'magic_xxz':
    param_name = 'delta'
    params = [np.round(i * 0.1 - 2, 1) for i in range(1, 40)]
    h_func = get_magic_xxz_dmrg_terms
elif model == 'magic_xxz_real_pairs':
    param_name = 'delta'
    params = [np.round(i * 0.1 - 2, 1) for i in range(1, 40)]
    h_func = get_magic_xxz_real_pairs_dmrg_terms
elif model == 't_ising':
    param_name = 'h'
    params = [np.round(i * 0.1, 1) for i in range(25)]
    h_func = get_t_ising_dmrg_terms
elif model == 'ising_magic':
    param_name = 'h'
    params = [np.round(i * 0.1, 1) for i in range(15, 20)]
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
    m2s = np.zeros(len(params))
    m2s_optimized = np.zeros(len(params))
    m2s_maximized = np.zeros(len(params))
    mhalves_optimized = np.zeros(len(params))
    mhalves_maximized = np.zeros(len(params))
    mhalves = np.zeros(len(params))
    best_bases = []
    worst_bases = []
    magnetization = np.zeros(len(params), dtype=complex)
    cicj = np.zeros(len(params), dtype=complex)
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
        curr = bops.contract(psi[int(n / 2)], psi[int(n / 2)], '0', '0*')
        for site in range(int(n / 2) + 1, n):
            curr = bops.contract(curr, psi[site], [2 * (site - int(n / 2)) - 1], '0')
            curr = bops.contract(curr, psi[site], [2 * (site - int(n / 2))], '0*')
        reorder = [2 * j for j in range(int(n/2))] + [2 * j + 1 for j in range(int(n/2) - 1)] + [n, n - 1, n + 1] #[0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 16, 15, 17]
        dm = curr.tensor.transpose(reorder).reshape([d ** int(n / 2), d ** int(n / 2)])
        mhalf = magicRenyi.getHalfRenyiExact_dm(dm, d)
        curr = bops.contract(psi[int(n / 2)], psi[int(n / 2)], '0', '0*')
        for site in range(int(n / 2) + 1, n):
            curr = bops.contract(curr, psi[site], [2 * (site - int(n / 2)) - 1], '0')
            curr = bops.contract(curr, psi[site], [2 * (site - int(n / 2))], '0*')
        mhalf_optimized, mhalf_best_basis, mhalf_maximized, mhalf_worst_basis = \
            magicRenyi.getHalfRenyiExact_dm_optimized(dm, d)
        magnetization[pi] = sum([bops.getExpectationValue(psi, [tn.Node(np.eye(2)) for i in range(j)] + \
                            [tn.Node(basic.pauli2Z)] + [tn.Node(np.eye(2)) for i in range(n - 1 - j)])  for j in range(n)])
                                 # bops.getExpectationValue(psi, [tn.Node(basic.pauli2Z) for site in range(n)])
        # cicj[pi] = sum([bops.getExpectationValue(psi,
        #                                          [tn.Node(np.eye(d)) for site in range(i)] + [tn.Node(np.array([[0, 1], [0, 0]]))] +
        #                                          [tn.Node(np.eye(d)) for site in range(j - i - 1)] + [tn.Node(np.array([[0, 0], [1, 0]]))] +
        #                                          [tn.Node(np.eye(d)) for site in range(n - j - 1)]) for i in range(n - 1) for j in range(i+1, n)]) + \
        #            sum([bops.getExpectationValue(psi,
        #         [tn.Node(np.eye(d)) for site in range(i)] + [tn.Node(np.array([[0, 0], [1, 0]]))] +
        #         [tn.Node(np.eye(d)) for site in range(j - i - 1)] + [tn.Node(np.array([[0, 1], [0, 0]]))] +
        #         [tn.Node(np.eye(d)) for site in range(n - j - 1)]) for i in range(n - 1) for j in range(i + 1, n)])
        print(psi[int(n/2)].tensor.shape)
        if psi[int(n / 2)].tensor.shape[0] > 4:
            psi_copy = bops.relaxState(psi, 4)
            print(bops.getOverlap(psi, psi_copy))
        mhalves[pi] = mhalf
        p2s[pi] = bops.getRenyiEntropy(psi, 2, int(n / 2))
        m2s[pi] = m2
        m2s_optimized[pi] = m2_optimized
        m2s_maximized[pi] = m2_maximized
        best_bases.append([phase / np.pi for phase in best_basis])
        worst_bases.append([phase / np.pi for phase in worst_basis])
        mhalves_optimized[pi] = mhalf_optimized
        mhalves_maximized[pi] = mhalf_maximized
        print(param)
        with open(filename(indir, model, param_name, param, n), 'wb') as f:
            pickle.dump([psi, m2, m2_optimized, best_basis, mhalf, m2_maximized, worst_basis,
                         mhalf_optimized, mhalf_best_basis, mhalf_maximized, mhalf_worst_basis], f)
    f, axs = plt.subplots(3, 1, gridspec_kw={'wspace':0, 'hspace':0}, sharex='all')
    axs[1].plot(params, m2s)
    axs[1].plot(params, m2s_optimized, '--k')
    axs[1].plot(params, m2s_maximized, ':k')
    axs[1].legend([r'$m_2$', r'$m_2$ optimized', r'$m_2$ maximized'])
    axs[2].plot(params, mhalves)
    axs[2].plot(params, mhalves_optimized, '--k')
    axs[2].plot(params, mhalves_maximized, ':k')
    axs[2].legend([r'$m_{1/2}$', r'$m_{1/2}$ optimized', r'$m_{1/2}$ maximized'])
    axs[0].plot(params, p2s)
    axs[0].plot(params, np.real(magnetization) / max(1, np.max(np.abs(magnetization))))
    # axs[0].plot(params, np.real(cicj) / max(1, np.max(np.abs(cicj))))
    axs[0].legend([r'$p_2$', r'$S_z / $' + str(np.round(max(1, np.max(np.abs(magnetization))), 1)), r'($c_i c_j^\dagger$ + h.c.)/' + str(np.round(np.real(np.max(np.abs(cicj))), 1))])
    print(best_bases)
    print(worst_bases)
    plt.title(model)
    plt.xlabel(param_name)
    b = 1
    plt.show()

run()

def analyze_scaling():
    ns = [8, 10, 12, 14, 16]
    Szs = np.zeros((len(ns), len(params)))
    p2s = np.zeros((len(ns), len(params)))
    m2s = np.zeros((len(ns), len(params)))
    m2s_maximized = np.zeros((len(ns), len(params)))
    m2s_optimized = np.zeros((len(ns), len(params)))
    mhalves = np.zeros((len(ns), len(params)))
    mhalves_maximized = np.zeros((len(ns), len(params)))
    mhalves_optimized = np.zeros((len(ns), len(params)))
    for pi in range(len(params)):
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
    f, axs = plt.subplots(3, 3, gridspec_kw={'wspace': 0, 'hspace': 0}, sharex='all')
    axs[0, 0].pcolormesh(Szs)
    axs[0, 1].pcolormesh(p2s)
    plt.show()

# analyze_scaling()