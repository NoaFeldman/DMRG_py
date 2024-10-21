import numpy as np
import tensornetwork as tn
from cv2.gapi import networks

import basicOperations as bops
# import matplotlib.pyplot as plt
import PEPS as peps
import pepsExpect as pe
import os
import pickle
import trg


def get_node(empty_coeff=0.0):
    d = 2
    ten = np.zeros((d, d, d, d), dtype=complex)
    ten[0, 0, 0, 0] = empty_coeff
    ten[0, 0, 0, 1] = 1
    ten[0, 0, 1, 0] = 1
    ten[0, 1, 0, 0] = 1
    ten[1, 0, 0, 0] = 1
    node = tn.Node(ten)
    return node


def bmps_boundaries(dirname: str, empty_coeff=0.0, steps=1000, chi=32):
    filename = dirname + '/bmps_dimer_chi_' + str(chi)
    if os.path.exists(filename):
        basic_node, cUp, dUp, cDown, dDown, leftRow, rightRow = pickle.load(open(filename, 'rb'))
        return basic_node, cUp, dUp, cDown, dDown, leftRow, rightRow
    node = get_node(empty_coeff)
    # double_node = tn.Node(bops.permute(bops.contract(bops.contract(node, node, '1', '3'),
    #                             bops.contract(node, node, '1', '3'), '15', '03'), [0, 2, 3, 6, 4, 7, 1, 5])\
    #                       .tensor.reshape([d**2]*4))

    upRow, downRow, leftRow, rightRow = peps.applyBMPS(node, node, d=d, steps=steps, chi=chi)

    [cUp, dUp, te] = bops.svdTruncation(upRow, [0, 1], [2, 3], '>>', maxBondDim=chi, maxTrunc=15)
    [cDown, dDown, te] = bops.svdTruncation(downRow, [0, 1], [2, 3], '>>', maxBondDim=chi, maxTrunc=15)
    pickle.dump([node, cUp, dUp, cDown, dDown, leftRow, rightRow], open(filename, 'wb'))
    return node, cUp, dUp, cDown, dDown, leftRow, rightRow


def expectation_value(basic_node: tn.Node, cUp: tn.Node, dUp: tn.Node, cDown: tn.Node, dDown: tn.Node, leftRow: tn.Node, rightRow: tn.Node, l):
    left = leftRow
    for i in range(int(l/2)):
        left = bops.contract(bops.contract(bops.contract(bops.contract(
            left, dDown, '0', '2'),
            basic_node, '04', '32'),
            basic_node, '03', '32'),
            cUp, '03', '01')
        left = bops.contract(bops.contract(bops.contract(bops.contract(
            left, cDown, '0', '2'),
            basic_node, '04', '32'),
            basic_node, '03', '32'),
            dUp, '03', '01')
    result = bops.contract(left, rightRow, '0123', '3210')
    return result.tensor * 1


def two_point_correlation(basic_node: tn.Node, cUp: tn.Node, dUp: tn.Node, cDown: tn.Node, dDown: tn.Node, leftRow: tn.Node, rightRow: tn.Node, l):
    d = basic_node[0].dimension
    proj = tn.Node(np.diag([1] + [0] * (d - 1)))
    left = bops.contract(bops.contract(bops.contract(bops.contract(
        leftRow, dDown, '0', '2'),
        basic_node, '04', '32'),
        bops.contract(basic_node, proj, '3', '0'), '03', '32'),
        cUp, '03', '01')
    left = bops.contract(bops.contract(bops.contract(bops.contract(
        left, cDown, '0', '2'),
        basic_node, '04', '32'),
        basic_node, '03', '32'),
        dUp, '03', '01')
    right = bops.contract(bops.contract(bops.contract(bops.contract(
        rightRow, dUp, '0', '2'),
        bops.permute(bops.contract(basic_node, proj, '1', '0'), [0, 3, 1, 2]), '04', '10'),
        basic_node, '03', '10'),
        cDown, '03', '01')
    right = bops.contract(bops.contract(bops.contract(bops.contract(
        right, cUp, '0', '2'),
        basic_node, '04', '10'),
        basic_node, '03', '10'),
        dDown, '03', '01')
    left_site = expectation_value(basic_node, cUp, dUp, cDown, dDown, left, rightRow, l - 2)
    right_site = expectation_value(basic_node, cUp, dUp, cDown, dDown, leftRow, right, l - 2)
    both = expectation_value(basic_node, cUp, dUp, cDown, dDown, left, right, l - 4)
    norm = expectation_value(basic_node, cUp, dUp, cDown, dDown, leftRow, rightRow, l)
    return both / norm - left_site*right_site / norm**2


def get_corr_length(dirname: str, empty_coeff=0.0, chi=8):
    filename = dirname + '/corrs_vs_ds_chi_' + str(chi)
    if os.path.exists(filename):
        [ds, corrs] = pickle.load(open(filename, 'rb'))
    else:
        basic_node, cUp, dUp, cDown, dDown, leftRow, rightRow = bmps_boundaries(dirname=dirname, empty_coeff=empty_coeff, chi=chi)
        ds = [i * 2 for i in range(3, 25)]
        corrs = np.round([two_point_correlation(basic_node, cUp, dUp, cDown, dDown, leftRow, rightRow, l) for l in ds], 15)
        pickle.dump([ds, corrs], open(filename, 'wb'))
    finite_ind = np.where(corrs < 1e-13)[0][0]
    xi, const = np.polyfit(ds[:finite_ind], np.log(np.abs(corrs[:finite_ind])), 1)
    return -xi


def get_kappa(dirname: str, empty_coeff=0.0):
    chis = [2**i for i in range(1, 11)]
    xis = [get_corr_length(dirname=dirname, empty_coeff=empty_coeff, chi=chi) for chi in chis]
    kappa, const = np.polyfit(np.log(xis), np.log(chis), 1)
    import matplotlib.pyplot as plt
    plt.plot(np.log(xis), np.log(chis))
    plt.plot(np.log(xis), kappa * np.log(xis) + const)
    plt.show()
    return kappa


def in_circle(ri, rj, r):
    return (r - 0.5 - ri)**2 + (r - 0.5 - rj)**2 < r**2


def count_nodes(r):
    res = 0
    for ri in range(2 * r):
        for rj in range(2 * r):
            if in_circle(ri, rj, r):
                res += 1
    return res


def edge_inds(ri, rj, r):
    res = []
    if not in_circle(ri - 1, rj, r):
        res.append(0)
    if not in_circle(ri, rj + 1, r):
        res.append(1)
    if not in_circle(ri + 1, rj, r):
        res.append(2)
    if not in_circle(ri, rj - 1, r):
        res.append(3)
    return res


def in_A(ri, rj, r, theta_start, theta):
    if rj == r:
        angle = np.pi / 2 if r - ri > 0 else np.pi * 3 / 2
    elif ri == r:
        angle = 0 if rj - r > 0 else np.pi
    else:
        if r - ri > 0 and rj - r > 0:
            angle = np.arctan((r - ri) / (rj - r))
        elif r - ri > 0 and rj - r < 0:
            angle = np.arctan(np.abs((rj - r) / (r - ri))) + np.pi / 2
        elif r - ri < 0 and rj - r < 0:
            angle = np.arctan(np.abs((r - ri) / (rj - r))) + np.pi
        else:
            angle = np.arctan(np.abs((rj - r) / (r - ri))) + 3 * np.pi / 2
    return (theta_start < angle and angle <= (theta_start + theta)) or \
        (theta_start < angle + 2 * np.pi and angle + 2 * np.pi <= (theta_start + theta))


d = 2
rdm_tensor = np.zeros((2, 2, 2, 2, 2, 2, 2, 2), dtype=float)
rdm_tensor[1, 1, 0, 0, 0, 0, 0, 0] = 1
rdm_tensor[0, 0, 1, 1, 0, 0, 0, 0] = 1
rdm_tensor[0, 0, 0, 0, 1, 1, 0, 0] = 1
rdm_tensor[0, 0, 0, 0, 1, 0, 0, 1] = 1
rdm_tensor[0, 0, 0, 0, 0, 1, 1, 0] = 1
rdm_tensor[0, 0, 0, 0, 0, 0, 1, 1] = 1
rdm_tensor = rdm_tensor.reshape([4] * 4)


def get_norm_network(r):
    edge_projector_tensor = np.zeros((4, 1))
    edge_projector_tensor[0, 0] = 1
    edge_projector = tn.Node(edge_projector_tensor)

    network = [[tn.Node(np.ones((1, 1, 1, 1))) for ri in range(r * 2)] for rj in range(r * 2)]
    for ri in range(2 * r):
        for rj in range(2 * r):
            if in_circle(ri, rj, r):
                network[ri][rj] = tn.Node(rdm_tensor)
                for edge_ind in edge_inds(ri, rj, r):
                    network[ri][rj] = bops.permute(bops.contract(network[ri][rj], edge_projector, [edge_ind], '0'),
                                                   list(range(edge_ind)) + [3] + list(range(edge_ind, 3)))
    return network


def get_purity_network(r, theta_start, theta):

    A_tensor = np.kron(rdm_tensor, rdm_tensor)
    A_node = tn.Node(A_tensor)

    B_tensor = A_tensor.reshape([2] * 16)\
        .transpose([1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12]).reshape([16] * 4)
    B_node = tn.Node(B_tensor)

    edge_projector_tensor = np.zeros((16, 1))
    edge_projector_tensor[0, 0] = 1
    edge_projector = tn.Node(edge_projector_tensor)

    network = [[tn.Node(np.ones((1, 1, 1, 1))) for ri in range(r*2)] for rj in range(r * 2)]
    for ri in range(2 * r):
        for rj in range(2 * r):
            if in_circle(ri, rj, r):
                if in_A(ri, rj, r, theta_start, theta):
                    network[ri][rj] = A_node
                else:
                    network[ri][rj] = B_node
                for edge_ind in edge_inds(ri, rj, r):
                    network[ri][rj] = bops.permute(bops.contract(network[ri][rj], edge_projector, [edge_ind], '0'),
                                                   list(range(edge_ind)) + [3] + list(range(edge_ind, 3)))
    return network


def get_network_exp(network):
    if len(network) == 3:
        rows = [bops.contract(bops.contract(network[i][0], network[i][1], '1', '3'), network[i][2], '4', '3')
                for i in range(3)]
        return bops.contract(bops.contract(rows[0], rows[1], '147', '035'), rows[2], '579', '035').tensor.reshape([1])
    else: # rows == 4
        rows = [bops.contract(bops.contract(bops.contract(
            network[i][0], network[i][1], '1', '3'), network[i][2], '4', '3'), network[i][3], '6', '3')
            for i in range(4)]
        return bops.contract(bops.contract(rows[0], rows[1], '1469', '0357'),
                      bops.contract(rows[2], rows[3], '1469', '0357'), [6, 8, 9, 11], [0, 2, 3, 4]).tensor.reshape([1])


# chis = [2, 4, 8, 16, 32, 64]
# tes = []
# for chi in chis:
#     network = get_norm_network(10)
#     trg_network, te = trg.trg(network, chi=chi)
#     print(chi, te)
#     tes.append(te)
# dbg = 1


import sys
dirname = sys.argv[1]
chi = int(sys.argv[2])
theta_start = float(sys.argv[3]) * np.pi
theta = float(sys.argv[4]) * np.pi
r = int(sys.argv[5])
calc_norm = bool(int(sys.argv[6]))

rdm_tensor /= 2.5**(1/4)

if calc_norm:
    network = get_norm_network(r)
    norm_trg_network, norm_te = trg.trg(network, chi=chi)
    norm = get_network_exp(norm_trg_network)
else:
    norm = 1
    norm_te = 0
print([r, chi, norm, norm_te, count_nodes(r)])
network = get_purity_network(r, theta_start, theta)
p2_trg_network, p2_te = trg.trg(network, chi=chi)
p2 = get_network_exp(p2_trg_network) / norm**2
print([norm, p2, norm_te, p2_te])
pickle.dump([norm, p2, norm_te, p2_te],
            open(dirname + '/p2_r_' + str(r) + '_theta_start_' + sys.argv[3] + '_theta_' + sys.argv[4] \
                 + '_chi_' + str(chi), 'wb'))
