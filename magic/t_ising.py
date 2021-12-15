import basicOperations as bops
import numpy as np
import DMRG as dmrg
import tensornetwork as tn
from typing import List
import pickle
import magic.basicDefs as basic
import matplotlib.pyplot as plt
import magic.magicRenyi as magicrenyi

n = 32
d = 2
psi0 = bops.getStartupState(n, d, mode='antiferromagnetic')
hs = [np.round(i * 0.1, 1) for i in range(11)]
p2s = np.zeros(len(hs))
m2s = np.zeros(len(hs))
for hi in range(len(hs)):
    h = hs[hi]
    onsite_terms = [h * basic.pauli2X for i in range(n)]
    neighbor_terms = [np.kron(basic.pauli2Z, basic.pauli2Z) for i in range(n-1)]
    gs, E0, truncErrs = dmrg.DMRG(psi0, onsite_terms, neighbor_terms, accuracy=1e-14)
    print(gs[int(n/2)].tensor.shape)
    if gs[int(n/2)].tensor.shape == 16:
        b = 1
    p2s[hi] = bops.getRenyiEntropy(gs, 2, int(n/2))
    m2s[hi] = magicrenyi.getSecondRenyi(gs, d)
plt.plot(hs, p2s)
plt.plot(hs, m2s)
plt.show()
