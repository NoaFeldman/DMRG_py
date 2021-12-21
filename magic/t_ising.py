import basicOperations as bops
import numpy as np
import DMRG as dmrg
import tensornetwork as tn
import pickle
import magic.basicDefs as basic
import matplotlib.pyplot as plt
import magic.magicRenyi as magicrenyi

n = 16
d = 2
psi0 = bops.getStartupState(n, d, mode='antiferromagnetic')
hs = [np.round(i * 0.1, 1) for i in range(6)] + [np.round(i * 0.05 + 0.5, 2) for i in range(21)] + \
    [np.round(i * 0.1 + 1.5, 1) for i in range(10)]
p2s = np.zeros(len(hs))
m2s = np.zeros(len(hs))
m2s_optimized = np.zeros(len(hs))
for hi in range(len(hs)):
    h = hs[hi]
    with open('/home/noa/PycharmProjects/DMRG_py/magic/results/t_ising/h_' + str(h), 'rb') as f:
        [gs, m2, irr, irrr] = pickle.load(f)
    # onsite_terms = [h * basic.pauli2X for i in range(n)]
    # neighbor_terms = [np.kron(basic.pauli2Z, basic.pauli2Z) for i in range(n-1)]
    # gs, E0, truncErrs = dmrg.DMRG(psi0, onsite_terms, neighbor_terms, accuracy=1e-14)
    print(gs[int(n/2)].tensor.shape)
    if gs[int(n/2)].tensor.shape == 16:
        b = 1
    p2s[hi] = bops.getRenyiEntropy(gs, 2, int(n/2))
    # m2 = magicrenyi.getSecondRenyi(gs, d)
    m2s[hi] = m2
    m2_optimized, best_basis = magicrenyi.getSecondRenyi_optimizedBasis(gs, d)
    m2s_optimized[hi] = m2_optimized
    print(h)
    with open('/home/noa/PycharmProjects/DMRG_py/magic/results/t_ising/h_' + str(h), 'wb') as f:
        pickle.dump([gs, m2, m2_optimized, best_basis], f)
plt.plot(hs, p2s)
plt.plot(hs, m2s)
plt.plot(hs, m2s_optimized)
plt.show()
