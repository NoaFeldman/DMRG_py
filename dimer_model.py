import numpy as np
import tensornetwork as tn
import basicOperations as bops
# import matplotlib.pyplot as plt
import PEPS as peps
import pepsExpect as pe
import os
import pickle


def bmps_boundaries(dirname: str, empty_coeff=0.0, steps=500, chi=32):
    filename = dirname + '/bmps_dimer_chi_' + str(chi)
    if os.path.exists(filename):
        basic_node, cUp, dUp, cDown, dDown, leftRow, rightRow = pickle.load(open(filename, 'rb'))
        return basic_node, cUp, dUp, cDown, dDown, leftRow, rightRow
    d = 2
    ten = np.zeros((d, d, d, d), dtype=complex)
    ten[0, 0, 0, 0] = empty_coeff
    ten[0, 0, 0, 1] = 1
    ten[0, 0, 1, 0] = 1
    ten[0, 1, 0, 0] = 1
    ten[1, 0, 0, 0] = 1
    node = tn.Node(ten)
    upRow, downRow, leftRow, rightRow = peps.applyBMPS(node, node, d=d, steps=steps, chi=chi)

    [cUp, dUp, te] = bops.svdTruncation(upRow, [0, 1], [2, 3], '>>', maxBondDim=chi)
    [cDown, dDown, te] = bops.svdTruncation(downRow, [0, 1], [2, 3], '>>', maxBondDim=chi)
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
    proj = tn.Node(np.diag([1, 0]))
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
    basic_node, cUp, dUp, cDown, dDown, leftRow, rightRow = bmps_boundaries(empty_coeff=empty_coeff, chi=chi)
    ds = [i * 2 for i in range(3, 20)]
    corrs = np.round([two_point_correlation(basic_node, cUp, dUp, cDown, dDown, leftRow, rightRow, l) for l in ds], 15)
    pickle.dump([ds, corrs], open(dirname + '/corrs_vs_ds_chi_' + str(chi), 'wb'))
    xi, const = np.polyfit(ds, np.log(np.abs(corrs)), 1)
    # plt.plot(np.array(ds), np.log(np.abs(corrs)))
    # plt.plot(np.array(ds), xi * np.array(ds) + const)
    # plt.show()
    return -xi


def get_kappa(empty_coeff=0.0):
    chis = [2**i for i in range(3, 9)]
    xis = [get_corr_length(empty_coeff=empty_coeff, chi=chi) for chi in chis]
    kappa, const = np.polyfit(np.log(xis), np.log(chis), 1)
    return kappa

get_kappa()