import basicOperations as bops
import numpy as np
import DMRG as dmrg
import tensornetwork as tn
from typing import List
import pickle
import magic.basicDefs as basic
import matplotlib.pyplot as plt
import magic.magicRenyi as magicrenyi

n = 16
d = 2
psi = bops.getStartupState(n, d, mode='antiferromagnetic')
deltas = [np.round(i * 0.1 - 2, 1) for i in range(1, 40)]
p2s = np.zeros(len(deltas))
m2s = np.zeros(len(deltas))
m2s_optimized = np.zeros(len(deltas))
m2s_maximized = np.zeros(len(deltas))
mhalves = np.zeros(len(deltas))
for di in range(len(deltas)):
    delta = deltas[di]
    if delta in [-1.0, 1.0]:
        psi = bops.getStartupState(n, d, mode='antiferromagnetic')
    try:
        with open('/home/noa/PycharmProjects/DMRG_py/magic/results/xxz/delta_' + str(delta), 'rb') as f:
            [psi, m2, m2_optimized, best_basis, mhalf, m2_maximized, worst_basis] = pickle.load(f)
    except FileNotFoundError or EOFError:
        onsite_terms = [0 * np.eye(d) for i in range(n)]
        neighbor_terms = [np.kron(basic.pauli2Z, basic.pauli2Z) * delta + \
                          np.kron(basic.pauli2X, basic.pauli2X) + \
                          np.kron(basic.pauli2Y, basic.pauli2Y) for i in range(n-1)]
        psi, E0, truncErrs = dmrg.DMRG(psi, onsite_terms, neighbor_terms, initial_bond_dim=4)
        if psi[int(n/2)].tensor.shape[0] > 4:
            psi_copy = bops.relaxState(psi, 4)
            print(bops.getOverlap(psi, psi_copy))
            m2 = magicrenyi.getSecondRenyi(psi_copy, d)
            m2_optimized, best_basis = magicrenyi.getSecondRenyi_optimizedBasis(psi_copy, d)
            m2_maximized, worst_basis = magicrenyi.getSecondRenyi_optimizedBasis(psi_copy, d, opt='max')
        else:
            m2 = magicrenyi.getSecondRenyi(psi, d)
            m2_optimized, best_basis = magicrenyi.getSecondRenyi_optimizedBasis(psi, d)
            m2_maximized, worst_basis = magicrenyi.getSecondRenyi_optimizedBasis(psi, d, opt='max')
        curr = bops.contract(psi[int(n / 2)], psi[int(n / 2)], '0', '0*')
        for site in range(int(n / 2) + 1, n):
            curr = bops.contract(curr, psi[site], [2 * (site - int(n / 2)) - 1], '0')
            curr = bops.contract(curr, psi[site], [2 * (site - int(n / 2))], '0*')
        reorder = [0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 16, 15, 17]
        dm = curr.tensor.transpose(reorder).reshape([d ** int(n / 2), d ** int(n / 2)])
        mhalf = magicrenyi.getHalfRenyiExact_dm(dm, d)
    mhalves[di] = mhalf
    print(psi[int(n/2)].tensor.shape)
    p2s[di] = bops.getRenyiEntropy(psi, 2, int(n/2))
    m2s[di] = m2
    m2s_optimized[di] = m2_optimized
    m2s_maximized[di] = m2_maximized
    print(delta)
    with open('/home/noa/PycharmProjects/DMRG_py/magic/results/xxz/delta_' + str(delta), 'wb') as f:
        pickle.dump([psi, m2, m2_optimized, best_basis, mhalf, m2_maximized, worst_basis], f)
plt.plot(deltas, p2s)
plt.plot(deltas, m2s)
plt.plot(deltas, m2s_optimized)
plt.plot(deltas, m2s_maximized)
plt.show()
