import numpy as np
import tensornetwork as tn
import PEPS as peps
import basicOperations as bops


d = 2


# Following Eq. (30) in https://arxiv.org/pdf/2010.13817.pdf
# (U_ccZ)^J U_cZ U_Z|0>
def getBoundaries(J=0, steps=50):
    site_tensor = np.zeros([d, d**3, d**3], dtype=complex)
    for i in range(d**3):
        site_tensor[0, 0, i] = 1
        site_tensor[1, 1 + 1*d + 1 * d**2, i] = 1
    site_tensor = site_tensor.reshape([d] * 7)
    # Add -1 signs to pairs of the form 1---1 to apply U_cZ
    # Add -1**J signs to triangles 1--1--1 to apply (U_ccZ)^J
    for i in range(d):
        for j in range(d):
            for k in range(d):
                site_tensor[1, 1, 1, 1, i, j, k] *= (-1)**(i + j + k + J * (i*j + j*k))

    site = tn.Node(site_tensor)
    siteEnv = tn.Node(bops.multiContraction(site, site, '0', '0*').tensor.
                      transpose([0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11]).reshape([d**2] * 6),
                      axis_names=['ul', 'ur', 'mr', 'dr', 'dl', 'dm'])

    gammaD = tn.Node(np.ones([1, d**2, d**2, 1], dtype=complex))
    gammaC = tn.Node(np.ones([1, d**2, d**2, 1], dtype=complex))
    lambdaD = tn.Node(np.ones(4))
    lambdaC = tn.Node(np.ones(4))
    envOp = bops.multiContraction(siteEnv, siteEnv, '2', '5')
    for i in range(steps):
        gammaD, lambdaD, gammaC, lambdaC = \
            peps.bmpsRowStep(gammaD, lambdaD, gammaC, lambdaC, envOp, lattice='triangular', chi=1)
        gammaC, lambdaC, gammaD, lambdaD = \
            peps.bmpsRowStep(gammaC, lambdaC, gammaD, lambdaD, envOp, lattice='triangular', chi=16)
    edgeOp = bops.contractDiag(gammaD, lambdaC.tensor, 0)
    [corner, edgeOp] = peps.bmpsTriangularCorner(edgeOp, siteEnv, siteEnv, steps)
    edgeOp.add_axis_names(['chi1', 'edge1', 'edge2', 'chi2'])
    return site, edgeOp, corner


def hexagon_dm(site: tn.Node, edgeOp: tn.Node, corner: tn.Node):
    site_dm = tn.Node(np.kron(site.tensor, np.conj(site.tensor)).reshape([d, d] + [d**2] * 6),
                      axis_names=['pup', 'pdown', 'ul', 'ur', 'mr', 'dr', 'dl', 'ml'])
    edge = bops.multiContraction(corner, edgeOp, '2', '0')
    edge.add_axis_names(['chi1', 'corner', 'edge1', 'edge2', 'chi2'])
    left = bops.multiContraction(bops.multiContraction(edge, edge, '4', '0'), edge, '7', '0')
    curr = bops.multiContraction(left, site_dm, '345', '267')
    curr = bops.multiContraction(curr, site_dm, [4, 5, 10, 3], '2367')
    curr = bops.multiContraction(curr, site_dm, '128', '672')
    curr = bops.multiContraction(curr, site_dm, [9, 5, 12], [2, 6, 7])
    curr = bops.multiContraction(curr, site_dm, [1, 7, 14], [2, 6, 7])
    curr = bops.multiContraction(curr, edge, [1, 16, 17], [0, 1, 2])
    curr = bops.multiContraction(curr, site_dm, [7, 12], [2, 7])
    curr = bops.multiContraction(curr, edge, [0, 7, 21, 20], [4, 3, 2, 1])
    curr = bops.multiContraction(curr, site_dm, [11, 12, 16, 8], [2, 3, 6, 7])
    curr = bops.multiContraction(curr, edge, [10, 17, 18, 13, 14], [0, 1, 2, 3, 4])
    b = 1


# TODO get corner with corner method. Consider both C and D edges.
# TODO Add DM tests. Add U_z to site tensor.
site, edgeOp, corner = getBoundaries(steps=1)
hexagon_dm(site, edgeOp, corner)