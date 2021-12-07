import numpy as np
import tensornetwork as tn
import PEPS as peps
import basicOperations as bops
import pepsExpect as pe
import pickle
import magic.magicRenyi as magic
import matplotlib.pyplot as plt

d = 2
explicit = False
if explicit:
    corner_site_tensor = np.ones((d, d, d, d, d, d, d, d, d), dtype=complex)
    corner_site = tn.Node(corner_site_tensor)
    corner_site_env = tn.Node(bops.permute(bops.multiContraction(corner_site, corner_site, '0', '0*'),
                            [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15]).tensor.reshape([d**2] * 8))
    middle_site_tensor = np.ones((d, d, d, d, d), dtype=complex)
    middle_site = tn.Node(middle_site_tensor)
    middle_site_env = tn.Node(bops.permute(bops.multiContraction(middle_site, middle_site, '0', '0*'),
                                   [0, 4, 1, 5, 2, 6, 3, 7]).tensor.reshape([d**2] * 4))
    env_op = bops.multiContraction(bops.multiContraction(
        middle_site_env, corner_site_env, '1', '7'), corner_site_env, '19', '12')
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


def getSiteTensors(J=0):
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
        basic_site, ul, '0', '1'), dl, '1', '1'), dr, '2', '1'), ur, '3', '1')
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


def save_boundaries(J=0):
    [A, B,
     boundary_top_left, boundary_top_right, boundary_bottom_left, boundary_bottom_right,
     boundary_top_edge, boundary_right_edge, boundary_bottom_edge, boundary_left_edge] = getSiteTensors(J)
    nonPhysicalLegs = 1
    GammaTensor = np.ones((nonPhysicalLegs, d ** 2, nonPhysicalLegs), dtype=complex)
    GammaC = tn.Node(GammaTensor, name='GammaC', backend=None)
    LambdaC = tn.Node(np.eye(nonPhysicalLegs) / np.sqrt(nonPhysicalLegs), backend=None)
    GammaD = tn.Node(GammaTensor, name='GammaD', backend=None)
    LambdaD = tn.Node(np.eye(nonPhysicalLegs) / np.sqrt(nonPhysicalLegs), backend=None)
    AEnv = pe.toEnvOperator(bops.multiContraction(A, A, '4', '4*'))
    BEnv = pe.toEnvOperator(bops.multiContraction(B, B, '4', '4*'))
    steps = 50
    GammaC, LambdaC, GammaD, LambdaD = peps.getBMPSRowOps(GammaC, LambdaC, GammaD, LambdaD, AEnv, BEnv, steps)
    openA = tn.Node(
        np.transpose(np.reshape(np.kron(A.tensor, A.tensor), [d ** 2, d ** 2, d ** 2, d ** 2, d, d]), [4, 0, 1, 2, 3, 5]))
    openB = tn.Node(
        np.transpose(np.reshape(np.kron(B.tensor, B.tensor), [d ** 2, d ** 2, d ** 2, d ** 2, d, d]), [4, 0, 1, 2, 3, 5]))
    cUp = bops.multiContraction(GammaC, LambdaC, '2', '0', isDiag2=True)
    dUp = bops.multiContraction(GammaD, LambdaD, '2', '0', isDiag2=True)
    upRow = bops.multiContraction(cUp, dUp, '2', '0')
    downRow = bops.copyState([upRow])[0]
    rightRow = peps.bmpsCols(upRow, downRow, AEnv, BEnv, steps, option='right', X=upRow)
    leftRow = peps.bmpsCols(upRow, downRow, AEnv, BEnv, steps, option='left', X=upRow)
    with open('results/union_jack_boundaries_J_' + str(J), 'wb') as f:
        pickle.dump([upRow, downRow, leftRow, rightRow, openA, openB, A, B], f)


def get_boundaries(J=0):
    with open('results/union_jack_boundaries_J_' + str(J), 'rb') as f:
        [up_row, down_row, left_row, right_row, openA, openB, A, B] = pickle.load(f)
        return [up_row, down_row, left_row, right_row, openA, openB, A, B]


# TODO move functions outside of union_jack (maybe square_lattice)
# TODO 3*3 dm?
# TODO Renyi 2 vs J, Renyi 1/2 vs J
def get_explicit_pure_state(w=2, h=3):
    n = 2 * w * h
    right_edge = [2 * (w - 1 + w * i) for i in range(h)]
    left_edge = [w * 2 * i for i in range(h)]
    outside = [i + 1 for i in right_edge] + [2 * i + 1 for i in range(w * (h - 1), w * h)]
    zh0 = np.array([(-1)**(bin(i).count("1")) * int(i not in outside) for i in range(2**n)])
    neighbors = []
    neighbors += [[2 * i, 2 * (i + w)] for i in range(w * (h - 1))]
    neighbors += [[2 * i, 2 * i + 1] for i in range(w * (h - 1)) if 2*i not in right_edge]
    neighbors += [[2 * i, 2 * i + 2] for i in range(w * h) if 2*i not in right_edge]
    neighbors += [[2 * i, 2 * i - 1] for i in range(w * (h - 1)) if 2*i not in left_edge]
    neighbors += [[2 * i, 2 * i - (2 * w - 1)] for i in range(w,  w * h) if 2*i not in right_edge]
    neighbors += [[2 * i, 2 * i - (2 * w + 1)] for i in range(w,  w * h) if 2*i not in left_edge]
    CZ = np.ones(2**n)
    for pair in neighbors:
        for i in range(2**n):
            CZ[i] *= (-1)**(int(i & 2**pair[0] > 0) * int(i & 2**pair[1] > 0))
    del neighbors
    czhz0 = np.matmul(np.diag(CZ), zh0)
    del zh0
    triangles = []
    triangles += [[2 * i, 2 * i + 1, 2 * i + 2] for i in range(w * (h - 1)) if 2*i not in right_edge]
    triangles += [[2 * i, 2 * i + 1, 2 * (i + w)] for i in range(w * (h - 1)) if 2*i not in right_edge]
    triangles += [[2 * i, 2 * i - 1, 2 * (i + w)] for i in range(w * (h - 1)) if 2*i not in left_edge]
    triangles += [[2 * i, 2 * i - (2 * w - 1), 2 * i + 2] for i in range(w,  w * h) if 2*i not in right_edge]
    CCZ = np.ones(2 ** n, dtype=complex)
    for tri in triangles:
        for i in range(2 ** n):
            CCZ[i] *= (-1) ** \
                      (int(i & 2 ** tri[0] > 0) * int(i & 2 ** tri[1] > 0) * int(i & 2 ** tri[2] > 0))
    # del triangles
    in_system = [i for i in range(2**n) if (i & sum([2^j for j in np.unique(outside)])) == 0]
    return czhz0[in_system], CCZ[in_system]


def get_second_renyi_pure(J):
    [mid_site, corner_site, boundary_top_left, boundary_top_right, boundary_bottom_left, boundary_bottom_right,
        boundary_top_edge, boundary_right_edge, boundary_bottom_edge, boundary_left_edge] = getSiteTensors(J)
    psi5 = bops.multiContraction(bops.multiContraction(bops.multiContraction(bops.multiContraction(
        boundary_top_left, mid_site, '0', '0'), boundary_bottom_left, '0', '0'),
        boundary_bottom_right, '0', '0'), boundary_top_right, '0', '0').tensor.reshape([d**5])
    return magic.getSecondRenyiExact_dm(np.outer(psi5, np.conj(psi5)), d)

J_step = 0.05
Js = [J_step * i for i in range(int(1/J_step) + 1)]
m2s_tn = np.zeros(len(Js))
for j in range(len(Js)):
    [mid_site, corner_site, boundary_top_left, boundary_top_right, boundary_bottom_left, boundary_bottom_right,
        boundary_top_edge, boundary_right_edge, boundary_bottom_edge, boundary_left_edge] = getSiteTensors(Js[j])
    psi5 = bops.contract(bops.contract(bops.contract(bops.contract(
        mid_site, boundary_top_left, '0', '0'), boundary_bottom_left, '0', '0'),
        boundary_bottom_right, '0', '0'), boundary_top_right, '0', '0').tensor.transpose([1, 0, 4, 2, 3]).reshape([d**5])
    dm5 = np.outer(psi5, np.conj(psi5))
    dm5 /= np.trace(dm5)
    m2s_tn[j] = magic.getSecondRenyiExact_dm(dm5, d)
plt.plot(Js, m2s_tn)
plt.show()