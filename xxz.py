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
os.chdir('/home/noa/PycharmProjects/DMRG_py')

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


t_gate = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])
hadamard = np.array([[1, 1, ], [1, -1]]) / np.sqrt(2)
rotated_t_gate = np.matmul(hadamard, np.matmul(t_gate, hadamard))
def get_magic_ising_dmrg_terms(h):
    onsite_terms = [- h * t_gate for i in range(n)]
    neighbor_terms = [- np.kron(basic.pauli2Z, basic.pauli2Z) for i in range(n - 1)]
    return onsite_terms, neighbor_terms

def get_magic_xxz_dmrg_terms(delta):
    onsite_terms = [0 * np.eye(d) for i in range(n)]
    neighbor_terms = [np.kron(t_gate, t_gate) * delta + \
                      np.kron(basic.pauli2X, basic.pauli2X) + \
                      np.kron(basic.pauli2Y, basic.pauli2Y) for i in range(n - 1)]
    return onsite_terms, neighbor_terms


n = 16
d = 2
psi = bops.getStartupState(n, d, mode='antiferromagnetic')
model = sys.argv[1]
indir = sys.argv[2]
if model == 'xxz':
    param_name = 'delta'
    params = [np.round(i * 0.1 - 2, 1) for i in range(1, 40)]
    h_func = get_xxz_dmrg_terms
elif model == 'magic_xxz':
        param_name = 'delta'
        params = [np.round(i * 0.1 - 2, 1) for i in range(1, 40)]
        h_func = get_magic_xxz_dmrg_terms
elif model == 't_ising':
    param_name = 'h'
    params = [np.round(i * 0.1, 1) for i in range(25)]
    h_func = get_t_ising_dmrg_terms
elif model == 'ising_magic':
    param_name = 'h'
    params = [np.round(i * 0.1, 1) for i in range(25)]
    h_func = get_magic_ising_dmrg_terms
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
        with open(indir + '/magic/results/' + model + '/' + param_name + '_' + str(param), 'rb') as f:
            [psi, m2, m2_optimized, best_basis, mhalf, m2_maximized, worst_basis] = pickle.load(f)
    except FileNotFoundError or EOFError:
        onsite_terms, neighbor_terms = h_func(param)
        psi, E0, truncErrs = dmrg.DMRG(psi, onsite_terms, neighbor_terms, initial_bond_dim=4)
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
        reorder = [0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 16, 15, 17]
        dm = curr.tensor.transpose(reorder).reshape([d ** int(n / 2), d ** int(n / 2)])
        mhalf = magicRenyi.getHalfRenyiExact_dm(dm, d)
    magnetization[pi] = bops.getExpectationValue(psi, [tn.Node(basic.pauli2Z) for site in range(n)])
    cicj[pi] = sum([bops.getExpectationValue(psi,
                                             [tn.Node(np.eye(d)) for site in range(i)] + [tn.Node(np.array([[0, 1], [0, 0]]))] +
                                             [tn.Node(np.eye(d)) for site in range(j - i - 1)] + [tn.Node(np.array([[0, 0], [1, 0]]))] +
                                             [tn.Node(np.eye(d)) for site in range(n - j - 1)]) for i in range(n - 1) for j in range(i+1, n)]) + \
               sum([bops.getExpectationValue(psi,
            [tn.Node(np.eye(d)) for site in range(i)] + [tn.Node(np.array([[0, 0], [1, 0]]))] +
            [tn.Node(np.eye(d)) for site in range(j - i - 1)] + [tn.Node(np.array([[0, 1], [0, 0]]))] +
            [tn.Node(np.eye(d)) for site in range(n - j - 1)]) for i in range(n - 1) for j in range(i + 1, n)])
    mhalves[pi] = mhalf
    print(psi[int(n/2)].tensor.shape)
    p2s[pi] = bops.getRenyiEntropy(psi, 2, int(n / 2))
    m2s[pi] = m2
    m2s_optimized[pi] = m2_optimized
    m2s_maximized[pi] = m2_maximized
    best_bases.append([phase / np.pi for phase in best_basis])
    worst_bases.append([phase / np.pi for phase in worst_basis])
    curr = bops.contract(psi[int(n / 2)], psi[int(n / 2)], '0', '0*')
    for site in range(int(n / 2) + 1, n):
        curr = bops.contract(curr, psi[site], [2 * (site - int(n / 2)) - 1], '0')
        curr = bops.contract(curr, psi[site], [2 * (site - int(n / 2))], '0*')
    reorder = [0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 16, 15, 17]
    dm = curr.tensor.transpose(reorder).reshape([d ** int(n / 2), d ** int(n / 2)])
    mhalf_optimized, mhalf_best_basis, mhalf_maximized, mhalf_worst_basis = \
        magicRenyi.getHalfRenyiExact_dm_optimized(dm, d)
    mhalves_optimized[pi] = mhalf_optimized
    mhalves_maximized[pi] = mhalf_maximized
    print(param)
    with open(indir + '/magic/results/' + model + '/' + param_name + '_' + str(param), 'wb') as f:
        pickle.dump([psi, m2, m2_optimized, best_basis, mhalf, m2_maximized, worst_basis,
                     mhalf_optimized, mhalf_best_basis, mhalf_maximized, mhalf_worst_basis], f)
        # pickle.dump([psi, m2, m2_optimized, best_basis, mhalf, m2_maximized, worst_basis], f)
plt.plot(params, p2s)
plt.plot(params, m2s / 6)
plt.plot(params, m2s_optimized / 6, '--k')
plt.plot(params, m2s_maximized / 6, ':k')
plt.plot(params, np.real(magnetization))
plt.plot(params, np.real(cicj) / 12, '--k')
plt.plot(params, mhalves)
plt.legend([r'$p_2$', r'$m_2/6$', r'$m_2/6$ optimized', r'$m_2/6$ maximized', 'magnetization', r'$c_i c_j^\dagger$ + h.c.', r'$m_{1/2}$'])
plt.show()
