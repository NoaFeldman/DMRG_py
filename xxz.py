import basicOperations as bops
import numpy as np
import DMRG as dmrg
import tensornetwork as tn
import pickle
import basicDefs as basic
import magicRenyi
import sys
import random
import randomUs as ru
from os import path


d = 2
model = sys.argv[1]
indir = sys.argv[2]
n = int(sys.argv[3])
range_i = int(sys.argv[4])
range_f = int(sys.argv[5])


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

def get_magic_xxz_rotations_dmrg_terms(theta):
    onsite_terms = [0 * np.eye(d) for i in range(n)]
    neighbor_terms = [np.kron(np.diag([1, np.exp(1j * np.pi * theta)]), np.diag([1, np.exp(1j * np.pi * theta)])) + \
                      np.kron(np.diag([1, np.exp(1j * np.pi * theta)]), np.diag([1, np.exp(1j * np.pi * theta)])).T.conj() + \
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


# Following https://doi.org/10.1088/1361-648x/ab8bf9
def get_kitaev_chain_dmrg_terms(params):
    Delta = 1
    mu = params[0]
    t = params[1]
    onsite_terms = [np.array(np.diag([0, 1])) * (-mu) for i in range(n)]
    neighbor_terms = [np.array([[0,     0,  0, Delta],
                                [0,     0, -t, 0    ],
                                [0,     -t, 0, 0    ],
                                [Delta, 0,  0, 0    ]]) for i in range(n - 1)]
    return onsite_terms, neighbor_terms


def get_fermion_tight_binding_dmrg_terms(mu):
    onsite_terms = [np.array(np.diag([0, 1])) * mu for i in range(n)]
    neighbor_terms = [np.array([[0, 0, 0, 0],
                                [0, 0, 1, 0],
                                [0, 1, 0, 0],
                                [0, 0, 0, 0]]) for i in range(n - 1)]
    return onsite_terms, neighbor_terms


def filename(indir, model, param_name, param, n):
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
elif model == 'magic_xxz_rotations':
    param_name = 'theta'
    resolution = 0.01
    params = [np.round(i * resolution, 3) for i in range(range_f - range_i)]
    h_func = get_magic_xxz_rotations_dmrg_terms
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
elif model == 'kitaev':
    param_name = 'mu_t'
    mu_range = list(range(8, 12, 2))
    t_range = list(range(-40, 40, 2))
    # mu_range = list(range(-40, 40, 2))
    # t_range = np.array(range(range_i, range_f, 2))
    params = [[np.round(mui * 0.1, 8), np.round(ti * 0.1, 8)] for mui in mu_range for ti in t_range]
    h_func = get_kitaev_chain_dmrg_terms
elif model == 'fermion_tight_binding':
    param_name = 'mu'
    params = [np.round(1.5 + 0.1 * i, 8) for i in range(11)]
    h_func = get_fermion_tight_binding_dmrg_terms

thetas = [t * np.pi for t in [0.0, 0.15, 0.25, 0.4]]
phis = [p * np.pi for p in [0.0, 0.15, 0.25, 0.4]]
etas = [e * np.pi for e in [0.0, 0.15, 0.25, 0.4]]


def run():
    psi = [tn.Node(np.array([np.sqrt(2), np.sqrt(2)]).reshape([1, 2, 1])) for i in range(n)] #bops.getStartupState(n, d)
    m2s = np.zeros((len(params), len(thetas), len(phis), len(etas)))
    mhalves = np.zeros((len(params), len(thetas), len(phis), len(etas)))
    for pi in range(len(params)):
        param = params[pi]
        final_file_name = filename(indir, model, param_name, param, n)
        if path.exists(final_file_name):
            with open(final_file_name, 'rb') as f:
                form = pickle.load(f)
            if len(form) == 4:
                continue
            else:
                psi_orig, m2, mhalf = form
                m2_avg = magicRenyi.getSecondRenyiAverage(psi, int(n / 2), d)
                print('m2_avg = ' + str(m2_avg))
                print(psi[int(n / 2)].tensor.shape)
                print(param)
                with open(filename(indir, model, param_name, param, n), 'wb') as f:
                    pickle.dump([psi_orig, m2, mhalf, m2_avg], f)
        else:
            onsite_terms, neighbor_terms = h_func(param)
            try:
                psi, E0, truncErrs = dmrg.DMRG(psi, onsite_terms, neighbor_terms, accuracy=1e-10)
            except TypeError:
                H = np.zeros((d**n, d**n), dtype=complex)
                for i in range(n):
                    H += np.kron(np.eye(d**i), np.kron(onsite_terms[i], np.eye(d**(n - i - 1))))
                for i in range(n - 1):
                    H += np.kron(np.eye(d**i), np.kron(neighbor_terms[i], np.eye(d**(n - i - 2))))
                evals, evecs = np.linalg.eigh(H)
                gs = evecs[:, np.argmin(evals)]
                curr = tn.Node(gs.reshape([1] + [d] * n + [1]))
                psi = []
                for i in range(n - 1):
                    [l, curr, te] = bops.svdTruncation(curr, [0, 1], list(range(2, len(curr.tensor.shape))), '>>')
                    psi.append(l)
                psi.append(curr)
                psi[-1].tensor /= bops.getOverlap(psi, psi)**0.5
            psi_orig = psi
            if psi[int(n / 2)].tensor.shape[0] > 4:
                psi = bops.relaxState(psi, 4)
                print(bops.getOverlap(psi, psi_orig))
            m2 = magicRenyi.getSecondRenyi(psi, d)
            print('m2 = ' + str(m2))
            dm = get_half_system_dm(psi)
            mhalf = magicRenyi.getHalfRenyiExact_dm(dm, d)
            print('mhalf = ' + str(mhalf))
            m2_avg = magicRenyi.getSecondRenyiAverage(psi, int(n / 2), d)
            print('m2_avg = ' + str(m2_avg))
            print(psi[int(n/2)].tensor.shape)
            print(param)
            with open(filename(indir, model, param_name, param, n), 'wb') as f:
                pickle.dump([psi_orig, m2, mhalf, m2_avg], f)


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


def darken(color, i):
    result = '#'
    for ci in range(1, len(color)):
        result += hex(int(color[ci], base=16) + i)[2] if color[ci] != 'f' else color[ci]
    return result


def get_color_gradient(color):
    result = [color]
    for i in range(len(thetas)):
        result.append(darken(color, 4*i))
    return result


def analyze_basis():
    f, axs = plt.subplots(3, 1, gridspec_kw={'wspace': 0, 'hspace': 0}, sharex='all')

    # with open(indir + '/magic/results/' + model + '/m2s', 'rb') as f:
    #     m2s = pickle.load(f)
    # with open(indir + '/magic/results/' + model + '/mhalves', 'rb') as f:
    #     mhalves = pickle.load(f)
    with open(indir + '/magic/results/' + model + 'm2s_' + param_name + 's_' + str(params[0]) + '_' + str(params[-1]), 'rb') as f:
        m2s = pickle.load(f)
    with open(indir + '/magic/results/' + model + 'mhalves_' + param_name + 's_' + str(params[0]) + '_' + str(params[-1]), 'rb') as f:
        mhalves = pickle.load(f)
    for ti in range(len(thetas)):
        u_theta = ru.getUTheta(thetas[ti] / np.pi, d=2)
        for ph_i in range(len(etas)):
            u_phi = ru.getUPhi(etas[ph_i] / np.pi, d=2)
            u = np.matmul(u_theta, u_phi)
            X = np.matmul(u, np.matmul(basic.pauli2X, np.conj(np.transpose(u))))
            Y = np.matmul(u, np.matmul(basic.pauli2Y, np.conj(np.transpose(u))))
            Z = np.matmul(u, np.matmul(basic.pauli2Z, np.conj(np.transpose(u))))
            H = np.matmul(u, np.matmul(t_z, np.conj(np.transpose(u))))
            f, axs = plt.subplots(3, 1, gridspec_kw={'wspace': 0, 'hspace': 0}, sharex='all')
            p2s = np.zeros(len(params))
            szs = np.zeros(len(params))
            sxs = np.zeros(len(params))
            sys = np.zeros(len(params))
            shs = np.zeros(len(params))
            shszs = np.zeros(len(params))
            for pi in range(len(params)):
                param = params[pi]
                with open(filename(indir, model, param_name, param, n), 'rb') as f:
                    [psi, m2, m2_optimized, best_basis, mhalf, m2_maximized, worst_basis,
                     mhalf_optimized, mhalf_best_basis, mhalf_maximized, mhalf_worst_basis] = pickle.load(f)
                p2s[pi] = bops.getRenyiEntropy(psi, 2, int(n / 2))
                szs[pi] = sum([bops.getExpectationValue(psi, [tn.Node(np.eye(2)) for i in range(j)] + \
                                                        [tn.Node(Z)] + \
                                                        [tn.Node(np.eye(2)) for i in range(n - 1 - j)]) for j in
                               range(n)])
                    # sum([bops.getExpectationValue(psi, [tn.Node(np.eye(2)) for i in range(j)] + \
                    #                                     [tn.Node(Z), tn.Node(Z)] + \
                    #                                     [tn.Node(np.eye(2)) for i in range(n - 2 - j)]) for j in
                    #            range(n - 1)])
                sxs[pi] = sum([bops.getExpectationValue(psi, [tn.Node(np.eye(2)) for i in range(j)] + \
                                                        [tn.Node(X)] + \
                                                        [tn.Node(np.eye(2)) for i in range(n - 1 - j)]) for j in
                               range(n)])
                    # sum([bops.getExpectationValue(psi, [tn.Node(np.eye(2)) for i in range(j)] + \
                    #                                     [tn.Node(X), tn.Node(X)] + \
                    #                                     [tn.Node(np.eye(2)) for i in range(n - 2 - j)]) for j in
                    #            range(n - 1)])
                sys[pi] = sum([bops.getExpectationValue(psi, [tn.Node(np.eye(2)) for i in range(j)] + \
                                                        [tn.Node(Y)] + \
                                                        [tn.Node(np.eye(2)) for i in range(n - 1 - j)]) for j in
                               range(n)])
                    # sum([bops.getExpectationValue(psi, [tn.Node(np.eye(2)) for i in range(j)] + \
                    #                                     [tn.Node(Y), tn.Node(Y)] + \
                    #                                     [tn.Node(np.eye(2)) for i in range(n - 2 - j)]) for j in
                    #            range(n - 1)])
                shs[pi] = sum([bops.getExpectationValue(psi, [tn.Node(np.eye(2)) for i in range(j)] + \
                                                        [tn.Node(H)] + \
                                                        [tn.Node(np.eye(2)) for i in range(n - 1 - j)]) for j in
                               range(n)])
                    # sum([bops.getExpectationValue(psi, [tn.Node(np.eye(2)) for i in range(j)] + \
                    #                                     [tn.Node(H), tn.Node(H)] + \
                    #                                     [tn.Node(np.eye(2)) for i in range(n - 2 - j)]) for j in
                    #            range(n - 1)])
                shszs[pi] = sum([bops.getExpectationValue(psi, [tn.Node(np.eye(2)) for i in range(j)] + \
                                                          [tn.Node(H), tn.Node(Z)] + \
                                                          [tn.Node(np.eye(2)) for i in range(n - 2 - j)]) for j in
                                 range(n - 1)])
            axs[0].plot(params, p2s)
            axs[0].plot(params, szs / n)
            axs[0].plot(params, sxs / n)
            axs[0].plot(params, sys / n, '--k')
            axs[0].plot(params, shs / n, ':k')
            axs[0].plot(params, shszs / n)
            axs[0].legend(['p2', 'szszs', 'sxsx', 'sysy', 'shsh', 'shsz'])

            for ei in range(len(etas)):
                axs[1].plot(params, m2s[:, ti, ph_i, ei])
                axs[2].plot(params, mhalves[:, ti, ph_i, ei])
            plt.legend([str(eta / np.pi) for eta in etas])
            plt.title(str(ti) + ' ' + str(ph_i))
            plt.show()
    plt.xlabel(param_name)
    plt.ylabel('r$m_2$')
    plt.show()


def analyze_kitaev_2d():
    import matplotlib.pyplot as plt
    ff, axs = plt.subplots(4, 1, gridspec_kw={'wspace': 0, 'hspace': 0}, sharex='all')
    m2s = np.zeros((len(mu_range), len(t_range)), dtype=complex)
    mhalves = np.zeros((len(mu_range), len(t_range)), dtype=complex)
    p2s = np.zeros((len(mu_range), len(t_range)), dtype=complex)
    Es = np.zeros((len(mu_range), len(t_range)), dtype=complex)
    for mui in range(len(mu_range)):
        for ti in range(len(t_range)):
            pi = mui * len(mu_range) + ti
            param = params[pi]
            mu = param[0]
            t = param[1]
            try:
                with open(filename(indir, model, param_name, param, n), 'rb') as f:
                    [psi, m2, mhalf] = pickle.load(f)
                    onsite_terms, neighbor_terms = get_kitaev_chain_dmrg_terms(param)
                    Es[mui, ti] = dmrg.stateEnergy(psi, dmrg.getDMRGH(n, onsite_terms, neighbor_terms, d=2))
                    m2s[mui, ti] = m2
                    mhalves[mui, ti] = mhalf
                    p2s[mui, ti] = bops.getRenyiEntropy(psi, 2, int(len(psi) / 2))
            except FileNotFoundError:
                pass
    pcm = axs[0].pcolormesh(mu_range, t_range, np.abs(Es), shading='auto')
    ff.colorbar(pcm, ax=axs[0])
    axs[0].set_ylabel('E')
    pcm = axs[1].pcolormesh(mu_range, t_range, np.abs(p2s), shading='auto')
    ff.colorbar(pcm, ax=axs[1])
    axs[1].set_ylabel(r'$p_2$')
    pcm = axs[2].pcolormesh(mu_range, t_range, np.abs(m2s), shading='auto')
    ff.colorbar(pcm, ax=axs[2])
    axs[2].set_ylabel(r'$M_2$')
    pcm = axs[3].pcolormesh(mu_range, t_range, np.abs(mhalves), shading='auto')
    ff.colorbar(pcm, ax=axs[3])
    axs[3].set_ylabel(r'$M_{1/2}$')
    plt.show()


def analyze_kitaev_ts():
    import matplotlib.pyplot as plt
    ff, axs = plt.subplots(3, 1, gridspec_kw={'wspace': 0, 'hspace': 0}, sharex='all')
    ns = [6, 8, 10, 12, 16]
    mu = 5.0
    ts = [p[1] for p in params]
    legends = []
    for n in ns:
        curr_ts = []
        p2s = []
        m2s = []
        mhalves = []
        for ti in range(len(ts)):
            t = ts[ti]
            try:
                with open(filename(indir, model, param_name, [mu, t], n), 'rb') as f:
                    [psi_orig, m2, mhalf] = pickle.load(f)
                    if psi_orig[int(n / 2)].tensor.shape[0] > 4:
                        psi = bops.relaxState(psi_orig, 4)
                        print(bops.getOverlap(psi, psi_orig))
                    curr_ts.append(t)
                    p2s.append(bops.getRenyiEntropy(psi, 2, int(len(psi) / 2)))
                    m2s.append(m2)
                    mhalves.append(mhalf)
            except FileNotFoundError:
                pass
        axs[0].scatter(curr_ts, p2s)
        axs[1].scatter(curr_ts, m2s)
        axs[2].scatter(curr_ts, mhalves)
        legends.append(str(n))
    plt.legend(legends)
    plt.show()


def analyze_thin():
    import matplotlib.pyplot as plt
    f, axs = plt.subplots(2, 1)
    p2s = np.zeros(len(params))
    m2s = np.zeros(len(params))
    for pi in range(len(params)):
        param = params[pi]
        with open(filename(indir, model, param_name, param, n), 'rb') as f:
            a = pickle.load(f)
            psi = a[0]
            m2 = a[1]
        p2s[pi] = bops.getRenyiEntropy(psi, 2, int(len(psi) / 2))
        m2s[pi] = m2
    axs[0].plot(params, p2s)
    axs[1].plot(params, m2s)
    plt.show()
run()