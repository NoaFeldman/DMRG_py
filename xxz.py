import os.path
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
from scipy import linalg


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
def get_magic_ising_dmrg_terms(theta):
    onsite_terms = [- np.diag([1, np.exp(1j * np.pi * theta)]) for i in range(n)]
    neighbor_terms = [- np.kron(basic.pauli2X, basic.pauli2X) for i in range(n - 1)]
    return onsite_terms, neighbor_terms

def get_magic_ising_h_2_dmrg_terms(theta):
    onsite_terms = [- 2 * np.diag([1, np.exp(1j * np.pi * theta)]) for i in range(n)]
    neighbor_terms = [- np.kron(basic.pauli2X, basic.pauli2X) for i in range(n - 1)]
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
    resolution = 0.001
    params = [np.round(i * resolution, 3) for i in range(range_i, range_f)]
    h_func = get_magic_xxz_rotations_dmrg_terms
elif model == 't_ising':
    param_name = 'h'
    params = [np.round(i * 0.1, 1) for i in range(range_i, range_f)]
    h_func = get_t_ising_dmrg_terms
elif model == 'ising_magic':
    param_name = 'theta'
    params = [np.round(i * 0.005, 8) for i in range(range_i, range_f)]
    h_func = get_magic_ising_dmrg_terms
elif model == 'ising_magic_h_2':
    param_name = 'theta'
    params = [np.round(i * 0.005, 8) for i in range(range_i, range_f)]
    h_func = get_magic_ising_h_2_dmrg_terms
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
    mu_range = list(range(10, 12, 2))
    t_range = list(range(-40, 40, 2))
    # mu_range = list(range(-40, 40, 2))
    # t_range = np.array(range(range_i, range_f, 2))
    params = [[np.round(mui * 0.1, 8), np.round(ti * 0.1, 8)] for mui in mu_range for ti in t_range]
    h_func = get_kitaev_chain_dmrg_terms
elif model == 'fermion_tight_binding':
    param_name = 'mu'
    params = [np.round(1.5 + 0.1 * i, 8) for i in range(11)]
    h_func = get_fermion_tight_binding_dmrg_terms

angle_step = 10
thetas = [np.round(i/angle_step, 3) for i in range(angle_step)]
phis = [np.round(i/angle_step, 3) for i in range(angle_step)]
etas = [np.round(i/angle_step, 3) for i in range(angle_step)]

def run():
    psi = [tn.Node(np.array([np.sqrt(2), np.sqrt(2)]).reshape([1, 2, 1])) for i in range(n)] #bops.getStartupState(n, d)
    m2s = np.zeros((len(params), len(thetas), len(phis), len(etas)))
    mhalves = np.zeros((len(params), len(thetas), len(phis), len(etas)))
    m2_avgs = np.zeros((len(params), len(thetas), len(phis), len(etas)))
    for pi in range(len(params)):
        param = params[pi]
        print(param)
        final_file_name = filename(indir, model, param_name, param, n)
        if path.exists(final_file_name):
            with open(final_file_name, 'rb') as f:
                form = pickle.load(f)
            if len(form) == 4:
                continue
            else:
                psi_orig, m2, mhalf = form
                if psi_orig[int(n / 2)].tensor.shape[0] > 4:
                    psi = bops.relaxState(psi_orig, 4)
                    print(bops.getOverlap(psi, psi_orig))
                m2_avg = magicRenyi.getSecondRenyiAverage(psi, int(n / 2), d)
                print('m2_avg = ' + str(m2_avg))
                print(psi[int(n / 2)].tensor.shape)
                print(param)
                with open(filename(indir, model, param_name, param, n), 'wb') as f:
                    pickle.dump([psi_orig, m2, mhalf, m2_avg], f)
        else:
            onsite_terms, neighbor_terms = h_func(param)
            try:
                raise TypeError
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
            for ti in range(len(thetas)):
                for pi in range(len(phis)):
                    for ei in range(len(etas)):
                        u = tn.Node(np.matmul(np.matmul(linalg.expm(1j * np.pi * thetas[ti] * X),
                                                        linalg.expm(1j * np.pi * phis[pi] * Z)),
                                                        linalg.expm(1j * np.pi * etas[ei] * X)))
                        psi_curr = [bops.permute(bops.contract(site, u, '1', '0'), [0, 2, 1]) for site in psi]
                        m2s[ti, pi, ei] = magicRenyi.getSecondRenyi(psi_curr, d)
                        print('m2 = ' + str(m2))
                        dm = get_half_system_dm(psi_curr)
                        mhalves[ti, pi, ei] = magicRenyi.getHalfRenyiExact_dm(dm, d)
                        print('mhalf = ' + str(mhalf))
                        m2_avgs[ti, pi, ei] = magicRenyi.getSecondRenyiAverage(psi_curr, int(n / 2), d)
                        print('m2_avg = ' + str(m2_avg))
                        print(psi[int(n/2)].tensor.shape)
                        print(param)
            with open(filename(indir, model, param_name, param, n), 'wb') as f:
                pickle.dump([psi_orig, m2s, mhalves, m2_avgs], f)


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


def analyze():
    import matplotlib.pyplot as plt
    f, axs = plt.subplots(4, 1)
    p2s = np.zeros(len(params))
    m2s = np.zeros(len(params))
    mhalves = np.zeros(len(params))
    m2s_avgs = np.zeros(len(params))
    for pi in range(len(params)):
        param = params[pi]
        if not os.path.exists(filename(indir, model, param_name, param, n)):
            continue
        with open(filename(indir, model, param_name, param, n), 'rb') as f:
            a = pickle.load(f)
            psi = a[0]
            m2 = a[1]
            mhalf = a[2]
            m2_avg = a[3]
        p2s[pi] = bops.getRenyiEntropy(psi, 2, int(len(psi) / 2))
        m2s[pi] = m2
        mhalves[pi] = mhalf
        m2s_avgs[pi] = m2_avg
    axs[0].plot(params, p2s)
    axs[0].set_ylabel(r'$p_2$')
    axs[1].plot(params, m2s)
    axs[1].set_ylabel(r'$M_2$')
    axs[2].plot(params, mhalves)
    axs[2].set_ylabel(r'$M_{1/2}$')
    axs[3].plot(params, m2s_avgs)
    axs[3].set_ylabel(r'$\overline{M_2}$')
    plt.xlabel(r'$\theta/\pi$')
    plt.show()


# Eq. (7) https://arxiv.org/pdf/2205.02247.pdf
def extract_alpha_beta(ns, param, plot=False):
    m2s = np.zeros(len(ns))
    mhalves = np.zeros(len(ns))
    file_exists = np.ones(len(ns))
    for ni in range(len(ns)):
        n = ns[ni]
        if os.path.exists(filename(indir, model, param_name, param, n)):
            data = pickle.load(open(filename(indir, model, param_name, param, n), 'rb'))
            print(n, len(data))
            m2s[ni] = data[1]
            mhalves[ni] = data[2]
        else:
            file_exists[ni] = 0
    curr_ns = np.array(ns)[np.where(file_exists != 0)[0][:]]
    curr_m2s = m2s[np.where(file_exists!=0)[0][:]]
    curr_mhalves = mhalves[np.where(file_exists!=0)[0][:]]
    coeff_2 = np.polyfit(curr_ns, curr_m2s, 1)
    coeff_half = np.polyfit(curr_ns, curr_mhalves, 1)
    if plot:
        import matplotlib.pyplot as plt
        plt.scatter(curr_ns, curr_m2s)
        plt.plot(curr_ns, coeff_2[0] * np.array(curr_ns) + coeff_2[1])
        plt.scatter(curr_ns, curr_mhalves)
        plt.plot(curr_ns, coeff_half[0] * np.array(curr_ns) + coeff_half[1])
        plt.title(str(param))
        plt.show()
    return list(coeff_2) + list(coeff_half)


def analyze_kitaev():
    ns = [8, 10, 12, 14, 16]
    p2s = np.zeros(len(params))
    alphas_2 = np.zeros(len(params))
    betas_2 = np.zeros(len(params))
    alphas_half = np.zeros(len(params))
    betas_half = np.zeros(len(params))
    for pi in range(len(params)):
        param = params[pi]
        alphas_2[pi], betas_2[pi], alphas_half[pi], betas_half[pi] = extract_alpha_beta(ns, param)
        p2s[pi] = bops.getRenyiEntropy(pickle.load(open(filename(indir, model, param_name, param, 12), 'rb'))[0], 2, 6)
    import matplotlib.pyplot as plt
    plt.plot([param[1] for param in params], p2s)
    plt.plot([param[1] for param in params], alphas_2)
    plt.plot([param[1] for param in params], betas_2)
    plt.plot([param[1] for param in params], alphas_half)
    plt.plot([param[1] for param in params], betas_half)
    plt.xlabel(r'$t$')
    plt.legend([r'$p_2$', r'$\alpha_2$', r'$\beta_2$', r'$\alpha_{1/2}$', r'$\beta_{1/2}$'])
    plt.show()


def analyze_ising_2d():
    n = 12
    import matplotlib.pyplot as plt
    thetas = [np.round(0.005 * ti, 8) for ti in range(int(2 / 0.005))]
    hs = [np.round(0.1 * hi, 8) for hi in range(30)]
    ff, axs = plt.subplots(4, 1)
    p2s = np.zeros((len(thetas), len(hs)))
    m2s = np.zeros((len(thetas), len(hs)))
    mhalves = np.zeros((len(thetas), len(hs)))
    m2_avgs = np.zeros((len(thetas), len(hs)))
    for hi in range(len(hs)):
        h = hs[hi]
        for ti in range(len(thetas)):
            theta = thetas[ti]
            try:
                [psi_orig, m2, mhalf, m2_avg] = pickle.load(open(filename(indir, model, 'theta_h', [theta, h], n), 'rb'))
            except FileNotFoundError:
                continue
            p2s[ti, hi] = bops.getRenyiEntropy(psi_orig, 2, int(n/2))
            m2s[ti, hi] = m2
            mhalves[ti, hi] = mhalf
            m2_avgs[ti, hi] = m2_avg
    axs[0].pcolormesh(hs, thetas, p2s)
    axs[0].set_title(r'$p_2$')
    axs[0].set_ylabel(r'$\theta/\pi$')
    axs[1].pcolormesh(hs, thetas, m2s)
    axs[1].set_title(r'$M_2$')
    axs[1].set_ylabel(r'$\theta/\pi$')
    axs[2].pcolormesh(hs, thetas, mhalves)
    axs[2].set_title(r'$M_{1/2}$')
    axs[2].set_ylabel(r'$\theta/\pi$')
    axs[3].pcolormesh(hs, thetas, m2_avgs)
    axs[3].set_title(r'$M_\bar{2}$')
    axs[3].set_ylabel(r'$\theta/\pi$')
    axs[3].set_xlabel(r'$h$')
    plt.show()
analyze_ising_2d()