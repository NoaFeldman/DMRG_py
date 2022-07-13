import numpy as np
import tensornetwork as tn
import DMRG as dmrg
import basicOperations as bops
import sys

# Eq. (1) https://journals.aps.org/prb/pdf/10.1103/PhysRevB.96.205104,
# but change z <--> x


c = np.array([[0, 1], [0, 0]])
Z = np.diag([1, -1])
X = np.array([[0, 1], [1, 0]])

d = 2

def ground_state(psi0, N, t, mu, h):
    hopping = np.kron(c.T, np.kron(X, np.kron(c, np.eye(d))))
    number = np.kron(np.matmul(c.T, c), np.eye(d))
    gauge_kinetic = np.kron(np.eye(d), np.kron(X, np.kron(np.eye(d), X)))
    gauge_kinetic_right_edge = np.eye(d**4)

    onsite_terms = [-mu * number for i in range(N)]
    neighbor_terms = [-t * hopping -h * gauge_kinetic for i in range(N)] + \
                    [-t * hopping -h * gauge_kinetic_right_edge]

    gs = dmrg.DMRG(psi0, onsite_terms, neighbor_terms, d=4)
    return gs


t = 1
N = 6
hs = [0.1 * hi for hi in range(-20, 21)]
mus = [0.1 * mui for mui in range(30)]
site_node = np.zeros([4, d**2, 4])
site_node[0, 0, 0] = 1
site_node[1, 0, 1] = 1
site_node[0, 3, 0] = 1
site_node[1, 3, 0] = 1
site_node[0, 1, 1] = 1
site_node[1, 2, 2] = 1
site_node[2, 2, 2] = 1
psi0 = [tn.Node(site_node[0, :, :].reshape([1, 4, 4]))] + [tn.Node(site_node) for i in range(N - 2)] + [tn.Node(site_node[:, :, 0].reshape([4, 4, 1]))]
psi0[-1].tensor /= np.sqrt(bops.getOverlap(psi0, psi0))

p2s = np.zeros((len(hs), len(mus)))
for mui in range(len(mus)):
    for hi in range(len(hs)):
        print(hi, mui)
        gs, E0, te = ground_state(psi0, N, t, mus[mui], hs[hi])
        p2s[hi, mui] = bops.getRenyiEntropy(gs, 2, int(N/2))
        # psi0 = gs
import matplotlib.pyplot as plt
plt.pcolormesh(hs, mus, p2s)
plt.show()