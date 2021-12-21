import numpy as np
import tensornetwork as tn
import PEPS as peps
import basicOperations as bops
import pepsExpect as pe
import pickle
import magic.magicRenyi as magic
import matplotlib.pyplot as plt
import magic.min_relative_entropy as dmin

d = 2
explicit = False
if explicit:
    corner_site_tensor = np.ones((d, d, d, d, d, d, d, d, d), dtype=complex)
    corner_site = tn.Node(corner_site_tensor)
    corner_env = tn.Node(bops.permute(bops.multiContraction(corner_site, corner_site, '0', '0*'),
                                      [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15]).tensor.reshape([d**2] * 8))
    middle_site_tensor = np.ones((d, d, d, d, d), dtype=complex)
    middle_site = tn.Node(middle_site_tensor)
    middle_site_env = tn.Node(bops.permute(bops.multiContraction(middle_site, middle_site, '0', '0*'),
                                   [0, 4, 1, 5, 2, 6, 3, 7]).tensor.reshape([d**2] * 4))
    env_op = bops.multiContraction(bops.multiContraction(
        middle_site_env, corner_env, '1', '7'), corner_env, '19', '12')
    gammaC = tn.Node(np.ones((1, d**2, d**2, d**2, 1), dtype=complex))
    gammaD = tn.Node(np.ones((1, d**2, d**2, d**2, 1), dtype=complex))
    lambdaC = tn.Node(np.ones(1, dtype=complex))
    lambdaD = tn.Node(np.ones(1, dtype=complex))
    gammaC, lambdaC, gammaD, lambdaD = \
        peps.bmpsRowStep(gammaC, lambdaC, gammaD, lambdaD, env_op, lattice='unionJack', chi=2)
    gammaD, lambdaD, gammaC, lambdaC = \
        peps.bmpsRowStep(gammaD, lambdaD, gammaC, lambdaC, env_op, lattice='unionJack', chi=2)
    C = bops.multiContraction(gammaC, lambdaC, '4', '0')
    D = bops.multiContraction(gammaD, lambdaD, '4', '0')


def getSiteTensors(J=0.0):
    ZH0 = np.array([1, -1]) / np.sqrt(2)
    basic_site = tn.Node(ZH0)
    squareCCZs = np.zeros((d, d, d, d, d, d, d, d, d, d), dtype=complex)
    for mid_i in range(d):
        for ul_i in range(d):
            for ur_i in range(d):
                for dl_i in range(d):
                    for dr_i in range(d):
                        squareCCZs[mid_i, mid_i, ul_i, ul_i, dl_i, dl_i, dr_i, dr_i, ur_i, ur_i] = (-1)**\
                            (mid_i * (ul_i + dl_i + dr_i + ur_i) + (ul_i + dr_i) * (dl_i + ur_i) + # CZ
                             J * mid_i * (ul_i + dr_i) * (dl_i + ur_i)) # CCZ
    squareOps = tn.Node(squareCCZs, axis_names=['m', 'm*', 'ul', 'ul*', 'dl', 'dl*', 'dr', 'dr*', 'ur', 'ur*'])
    [l, ur, te] = bops.svdTruncation(squareOps, list(range(8)), list(range(8, 10)), '<<')
    l = bops.permute(l, [0, 1, 8] + list(range(2, 8)))
    [l, dr, te] = bops.svdTruncation(l, list(range(7)), list(range(7, 9)), '<<')
    l = bops.permute(l, [0, 1, 7] + list(range(2, 7)))
    [l, dl, te] = bops.svdTruncation(l, list(range(6)), list(range(6, 8)), '<<')
    l = bops.permute(l, [0, 1, 6] + list(range(2, 6)))
    [mid, ul, te] = bops.svdTruncation(l, list(range(5)), list(range(5, 7)), '<<')
    mid = bops.permute(mid, [0, 1, 5] + list(range(2, 5)))
    # mid site indices: ul, dl, dr, ur, physical
    mid_site = bops.permute(bops.multiContraction(basic_site, mid, '0', '0'), [1, 2, 3, 4, 0])
    corner_site = bops.multiContraction(bops.multiContraction(bops.multiContraction(bops.multiContraction(
        basic_site, dr, '0', '1'), ur, '1', '1'), ul, '2', '1'), dl, '3', '1')
    boundary_top_left = bops.multiContraction(basic_site, ul, '0', '1')
    boundary_top_right = bops.multiContraction(basic_site, ur, '0', '1')
    boundary_bottom_left = bops.multiContraction(basic_site, dl, '0', '1')
    boundary_bottom_right = bops.multiContraction(basic_site, dr, '0', '1')
    boundary_top_edge = bops.multiContraction(bops.multiContraction(basic_site, dl, '0', '1'), dr, '1', '1')
    boundary_right_edge = bops.multiContraction(bops.multiContraction(basic_site, ul, '0', '1'), dl, '1', '1')
    boundary_bottom_edge = bops.multiContraction(bops.multiContraction(basic_site, ur, '0', '1'), ul, '1', '1')
    boundary_left_edge = bops.multiContraction(bops.multiContraction(basic_site, dr, '0', '1'), ur, '1', '1')
    return [mid_site, corner_site,
            boundary_top_left, boundary_top_right, boundary_bottom_left, boundary_bottom_right,
            boundary_top_edge, boundary_right_edge, boundary_bottom_edge, boundary_left_edge]


def save_boundaries(J=0.0):
    [mid_site, corner_site,
     boundary_top_left, boundary_top_right, boundary_bottom_left, boundary_bottom_right,
     boundary_top_edge, boundary_right_edge, boundary_bottom_edge, boundary_left_edge] = getSiteTensors(J)
    nonPhysicalLegs = 1
    GammaTensor = np.ones((nonPhysicalLegs, d ** 2, nonPhysicalLegs), dtype=complex)
    GammaC = tn.Node(GammaTensor, name='GammaC', backend=None)
    LambdaC = tn.Node(np.eye(nonPhysicalLegs) / np.sqrt(nonPhysicalLegs), backend=None)
    GammaD = tn.Node(GammaTensor, name='GammaD', backend=None)
    LambdaD = tn.Node(np.eye(nonPhysicalLegs) / np.sqrt(nonPhysicalLegs), backend=None)
    mid_site_env = pe.toEnvOperator(bops.multiContraction(mid_site, mid_site, '4', '4*'))
    corner_site_env = pe.toEnvOperator(bops.multiContraction(corner_site, corner_site, '4', '4*'))
    steps = 50
    GammaC, LambdaC, GammaD, LambdaD = peps.getBMPSRowOps(GammaC, LambdaC, GammaD, LambdaD, mid_site_env, corner_site_env, steps)
    open_mid = tn.Node(
        np.transpose(np.reshape(np.kron(mid_site.tensor, mid_site.tensor), [d ** 2, d ** 2, d ** 2, d ** 2, d, d]), [4, 0, 1, 2, 3, 5]))
    open_corner = tn.Node(
        np.transpose(np.reshape(np.kron(corner_site.tensor, corner_site.tensor), [d ** 2, d ** 2, d ** 2, d ** 2, d, d]), [4, 0, 1, 2, 3, 5]))
    cUp = bops.multiContraction(GammaC, LambdaC, '2', '0', isDiag2=True)
    dUp = bops.multiContraction(GammaD, LambdaD, '2', '0', isDiag2=True)
    upRow = bops.multiContraction(cUp, dUp, '2', '0')
    downRow = bops.copyState([upRow])[0]
    rightRow = peps.bmpsCols(upRow, downRow, mid_site_env, corner_site_env, steps, option='right', X=upRow)
    leftRow = peps.bmpsCols(upRow, downRow, mid_site_env, corner_site_env, steps, option='left', X=upRow)
    with open('results/union_jack_boundaries_J_' + str(J), 'wb') as f:
        pickle.dump([upRow, downRow, leftRow, rightRow, open_mid, open_corner, mid_site, corner_site], f)


def get_boundaries(J=0.0):
    with open('results/union_jack_boundaries_J_' + str(J), 'rb') as f:
        [up_row, down_row, left_row, right_row, open_mid, open_corner, mid_site, corner_site] = pickle.load(f)
        return [up_row, down_row, left_row, right_row, open_mid, open_corner, mid_site, corner_site]


# TODO move functions outside of union_jack (maybe square_lattice)
# TODO 3*3 dm?
# TODO Renyi 2 vs J, Renyi 1/2 vs J
def get_second_renyi_pure(J):
    [mid_site, corner_site, boundary_top_left, boundary_top_right, boundary_bottom_left, boundary_bottom_right,
        boundary_top_edge, boundary_right_edge, boundary_bottom_edge, boundary_left_edge] = getSiteTensors(J)
    psi5 = bops.multiContraction(bops.multiContraction(bops.multiContraction(bops.multiContraction(
        boundary_top_left, mid_site, '0', '0'), boundary_bottom_left, '0', '0'),
        boundary_bottom_right, '0', '0'), boundary_top_right, '0', '0').tensor.reshape([d**5])
    return magic.getSecondRenyiExact_dm(np.outer(psi5, np.conj(psi5)), d)


def get_mixed_dm(J=0.):
    [up_row, down_row, left_row, right_row, open_mid, open_corner, mid_site, corner_site] = get_boundaries(J)
    mid_env = pe.toEnvOperator(bops.multiContraction(mid_site, mid_site, '4', '4*'))
    corner_env = pe.toEnvOperator(bops.multiContraction(corner_site, corner_site, '4', '4*'))
    left = bops.contract(bops.contract(down_row,
                                       bops.contract(left_row, left_row, '3', '0'), '3', '0'), up_row, '7', '0')
    left = bops.contract(bops.contract(bops.contract(bops.contract(
        left, corner_env, '23', '21'), mid_env, '82', '21'),
        corner_env, '82', '21'), mid_env, '823', '210')
    left = bops.contract(bops.contract(bops.contract(bops.contract(
        left, mid_env, '14', '21'), open_corner, '63', '32'),
        mid_env, '73', '21'), corner_env, '841', '210')
    left = bops.contract(down_row, bops.contract(up_row, left, '0', '1'), '3', '3')
    left = bops.contract(bops.contract(bops.contract(bops.contract(
        left, open_corner, '26', '32'), open_mid, [11, 6], '32'),
        open_corner, [13, 7], '32'), mid_env, [15, 7, 2], '321')
    left = bops.contract(bops.contract(bops.contract(bops.contract(
        left, mid_env, '17', '21'), open_corner, [14, 8], '32'),
        mid_env, [15, 10], '21'), corner_env, [16, 11, 1], '321')
    right = bops.contract(right_row, right_row, '3', '0')
    dm = bops.contract(left, right, [1, 15, 14, 12, 10, 0], '012345').tensor. \
        transpose([0, 2, 4, 6, 8, 1, 3, 5, 7, 9]).reshape([32, 32])
    dm /= np.trace(dm)
    dm = np.round(dm, 10)
    return dm

run = True
if run:
    J_step = 0.005
    Js = [J_step * i for i in range(int(1/J_step) + 1)]
    m05s_mixed = np.zeros(len(Js))
    for j in range(len(Js)):
        J = Js[j]
        # save_boundaries(J)
        dm = get_mixed_dm(J)
        dmin.get_robustness(dm)
        m05s_mixed[j] = magic.getHalfRenyiExact_dm(dm, d)


    m05s_pure = np.zeros(len(Js))
    m2s_pure = np.zeros(len(Js))
    for j in range(len(Js)):
        [mid_site, corner_site, boundary_top_left, boundary_top_right, boundary_bottom_left, boundary_bottom_right,
            boundary_top_edge, boundary_right_edge, boundary_bottom_edge, boundary_left_edge] = getSiteTensors(Js[j])
        psi5 = bops.contract(bops.contract(bops.contract(bops.contract(
            mid_site, boundary_top_left, '0', '0'), boundary_bottom_left, '0', '0'),
            boundary_bottom_right, '0', '0'), boundary_top_right, '0', '0').tensor.transpose([1, 0, 4, 2, 3]).reshape([d**5])
        dm5 = np.outer(psi5, np.conj(psi5))
        dm5 /= np.trace(dm5)
        m05s_pure[j] = magic.getHalfRenyiExact_dm(dm5, d)
        m2s_pure[j] = magic.getSecondRenyiExact_dm(dm5, d)
    plt.plot(Js, m05s_pure)
    plt.plot(Js, m2s_pure)
    with open('magic/results/union_jack_5_pure', 'wb') as f:
        pickle.dump([Js, m05s_pure], f)
    plt.plot(Js, m05s_mixed)
    with open('magic/results/union_jack_5_mixed', 'wb') as f:
        pickle.dump([Js, m05s_mixed], f)
    plt.show()