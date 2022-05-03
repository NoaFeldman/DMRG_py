import numpy as np
import tensornetwork as tn
import basicOperations as bops
import pepsExpect as pe
import PEPS as peps
import pickle
import randomUs as ru
from typing import List
import matplotlib.pyplot as plt
import os
import sys


def toric_code(g=0.0):
    #
    #    O
    #    |
    #  __*__O
    #    |
    # bond left, bond down, * in, O up in, O down in, * out, O up out, O down out
    tensor_op_bottom_left = np.zeros((2, 2, 1, 2, 2, 1, 2, 2), dtype=complex)
    for i in range(2):
        for j in range(2):
            # identity
            tensor_op_bottom_left[0, 0, 0, i, j, 0, i, j] = 1
            # X
            tensor_op_bottom_left[1, 1, 0, i, j, 0, i ^ 1, j ^ 1] = 1
    # |
    # O
    # bond up, O in, O out
    tensor_op_top = np.zeros((2, 2, 2), dtype=complex)
    for i in range(2):
        tensor_op_top[0, i, i] = 1
        tensor_op_top[1, i, i ^ 1] = 1
    # O--
    # bond left, O in, O out
    tensor_op_right = np.zeros((2, 2, 2), dtype=complex)
    for i in range(2):
        tensor_op_right[0, i, i] = 1
        tensor_op_right[1, i, i ^ 1] = 1

    op_bottom_left = tn.Node(tensor_op_bottom_left)
    op_top = tn.Node(tensor_op_top)
    op_right = tn.Node(tensor_op_right)

    tensor_base_site = np.zeros((1, 2, 2), dtype=complex)
    tensor_base_site[0, 0, 0] = 1
    base_site = tn.Node(tensor_base_site)

    #     0
    #     |
    # 3--- ---1
    #     |
    #     2
    A = bops.permute(bops.contract(bops.contract(bops.contract(
        base_site, op_bottom_left, '012', '234'), op_top, '3', '1'), op_right, '3', '1'), [3, 5, 1, 0, 2, 4, 6])
    B = bops.permute(bops.contract(bops.contract(bops.contract(
        base_site, op_top, '1', '1'), op_right, '1', '1'), op_bottom_left, '024', '234'), [0, 1, 3, 2, 4, 5, 6])

    single_site_tension = tn.Node(np.diag([1, 1+g]))
    A = bops.contract(bops.contract(A, single_site_tension, '5', '0'), single_site_tension, '5', '0')
    B = bops.contract(bops.contract(B, single_site_tension, '5', '0'), single_site_tension, '5', '0')

    A = tn.Node(A.tensor.reshape([2, 2, 2, 2, 4]))
    B = tn.Node(B.tensor.reshape([2, 2, 2, 2, 4]))

    AEnv = pe.toEnvOperator(bops.multiContraction(A, A, '4', '4*'))
    BEnv = pe.toEnvOperator(bops.multiContraction(B, B, '4', '4*'))
    upRow, downRow, leftRow, rightRow, openA, openB = peps.applyBMPS(A, B, d=4)
    with open('results/toricBoundaries_gauge_' + str(g), 'wb') as f:
        pickle.dump([upRow, downRow, leftRow, rightRow, openA, openB, A, B], f)
    circle = bops.contract(bops.contract(bops.contract(
        upRow, rightRow, '3', '0'), downRow, '5', '0'), leftRow, '70', '03')
    M = bops.contract(bops.contract(bops.contract(bops.contract(
        circle, openB, '07', '14'), openA, '017', '124'), openA, '523', '134'), openB, '5018', '1234')
    mat = np.round(np.real(M.tensor.transpose([0, 2, 4, 6, 1, 3, 5, 7]). reshape([4**4, 4**4]) / 0.03125), 10)
    b = 1

large_gs = [2.0, 3.0, 4.0]
for g in large_gs:
    toric_code(g)

# w and h are the dimensions of the system *tensor-wise* and not site-wise (when each tensor is two sites).
# list_of_sectors is a list of lists, from left to write and top to bottom.
# For a w*h system, the first w lists are of length 1+h (including edges).
# The last list is of length h, only right edges.
def get_local_vectors(w, h, list_of_sectors, free_site_choices):
    int_results = np.zeros((2 * w, h))

    # left edge sectors
    int_results[0, 0] = 1 if list_of_sectors[0][0] == 1 else -1
    for ri in range(h - 1):
        int_results[1, ri] = free_site_choices.pop()
        int_results[0, ri + 1] = list_of_sectors[0][ri + 1]
    int_results[1, h - 1] = list_of_sectors[1][h]

    # bulk columns
    for ci in range(1, w):
        int_results[2 * ci, 0] = list_of_sectors[ci + 1][0] / (int_results[2 * (ci - 1), 0] * int_results[2 * ci - 1, 0])
        for ri in range(h - 1):
            if ci == w - 1:
                if ri == 0:
                    int_results[2 * ci + 1, ri] = list_of_sectors[ci + 1][0] / int_results[2 * ci, 0]
                else:
                    int_results[2 * ci + 1, ri] = list_of_sectors[ci + 1][ri] / (int_results[2 * ci, ri] *
                                                                                 int_results[2 * ci + 1, ri - 1])
            else:
                int_results[2 * ci + 1, ri] = free_site_choices.pop()
            int_results[2 * ci, ri + 1] = list_of_sectors[ci][ri + 1] / \
                                          (int_results[2 * ci - 1, ri] *
                                           int_results[2 * ci - 1, ri + 1] *
                                           int_results[2 * ci - 2, ri + 1])
        int_results[2 * ci + 1, h - 1] = list_of_sectors[ci + 1][-1]
    results = [None for i in range(w * h)]
    for wi in range(w):
        for hi in range(h):
            top_vec = up_spin if int_results[2 * wi, hi] == 1 else down_spin
            right_vec = up_spin if int_results[2 * wi + 1, hi] == 1 else down_spin
            curr = np.kron(top_vec, right_vec)
            results[hi + wi * h] = curr
    return results


up_spin = np.array([1, 0]) * np.sqrt(2)
down_spin = np.array([0, 1]) * np.sqrt(2)
def get_random_local_vectors(w, h, list_of_sectors, bulk_ints=None) -> List[np.array]:
    if bulk_ints is None:
        bulk_ints = [np.random.randint(2) * 2 - 1 for i in range((h-1) * (w - 1))]
    results = get_local_vectors(w, h, list_of_sectors, bulk_ints)
    return results


def get_boundary_sectors(w, h, boundary_identifier):
    # After all of the boundary sections but one are set, the last one is determined by them
    boundary_length = 2 * (w + h - 1)
    boundary_sectors = [int(c) * 2 - 1 for c in bin(boundary_identifier).split('b')[1]]
    boundary_sectors = [-1] * (boundary_length - len(boundary_sectors)) + boundary_sectors
    boundary_sectors.append(np.prod(boundary_sectors))
    return boundary_sectors

# boundary_sectors are the sector eigenvalues on the boundary from the top left corner and down the left edge,
# then top-bottom for the middle, and then top to bottom in the right edge.
def get_list_of_sectors(w, h, boundary_identifier):
    boundary_sectors = get_boundary_sectors(w, h, boundary_identifier)
    res = [[boundary_sectors[i] for i in range(h)]]
    for i in range(1, w):
        res.append([boundary_sectors[h - 2 + i * 2]] + [1] * (h - 1) + [boundary_sectors[h - 1 + i * 2]])
    res.append([boundary_sectors[i] for i in range(-(h+1), 0)])
    return res


def get_random_operators(w, h, n, list_of_sectors, bulk_ints=None):
    random_local_vectors = [None for ni in range(n)]
    for ni in range(n):
        if bulk_ints is None:
            random_local_vectors[ni] = get_random_local_vectors(w, h, list_of_sectors)
        else:
            random_local_vectors[ni] = get_random_local_vectors(w, h, list_of_sectors, bulk_ints[ni])
    results = [None for ni in range(n)]
    for ni in range(n):
        results[ni] = [tn.Node(np.outer(random_local_vectors[ni][si], random_local_vectors[(ni + 1) % n][si]))
                       for si in range(w * h)]
    return results


def get_explicit_block(w, h, g, boundary_identifier=0):
    num_of_free_site_choices = (h - 1) * (w - 1)
    block_size = 2**num_of_free_site_choices
    ops = [[None for i in range(block_size)] for j in range(block_size)]
    list_of_sectors = get_list_of_sectors(w, h, boundary_identifier)
    for i in range(block_size):
        for j in range(i, block_size):
            choices_i = [int(c) * 2 - 1 for c in bin(i).split('b')[1]]
            choices_i = [-1] * (num_of_free_site_choices - len(choices_i)) + choices_i
            choices_j = [int(c) * 2 - 1 for c in bin(j).split('b')[1]]
            choices_j = [-1] * (num_of_free_site_choices - len(choices_j)) + choices_j
            ops[i][j] = get_random_operators(w, h, 2, list_of_sectors, [choices_i, choices_j])[0]

    with open('results/toricBoundaries_gauge_' + str(np.round(g, 8)), 'rb') as f:
        [upRow, downRow, leftRow, rightRow, openA, openB, A, B] = pickle.load(f)
    [cUp, dUp, te] = bops.svdTruncation(upRow, [0, 1], [2, 3], '>>')
    [cDown, dDown, te] = bops.svdTruncation(downRow, [0, 1], [2, 3], '>>')
    norm = pe.applyLocalOperators(cUp, dUp, cDown, dDown, leftRow, rightRow, A, B, w, h,
                                  [tn.Node(np.eye(4)) for i in range(w * h)])
    leftRow = bops.multNode(leftRow, 1 / norm ** (2 / w))

    res = np.zeros((block_size, block_size), dtype=complex)
    for i in range(block_size):
        for j in range(i, block_size):
            res[i, j] = pe.applyLocalOperators(cUp, dUp, cDown, dDown, leftRow, rightRow, A, B, w, h,
                                  ops[i][j])
            res[j, i] = np.conj(res[i, j])
    res /= 2**(2*w*h)
    return res


colors = ['#0000FF', '#9D02D7', '#EA5F94', '#FA8775', '#FFB14E', '#FFD700']
gs = [np.round(0.1 * G, 8) for G in range(11)]
plot_2_by_2 = False
def binary_string(i, N):
    curr = bin(i).split('b')[1]
    return '0' * (N - len(curr)) + curr
if plot_2_by_2:
    bs = np.array(range(2**6))
    import matplotlib.pyplot as plt
    p2s_all = []
    p1s_all = []
    for b in bs:
        p2s = np.zeros(len(gs))
        p1s = np.zeros(len(gs))
        for gi in range(len(gs)):
            block = get_explicit_block(2, 2, gs[gi], boundary_identifier=b)
            p2s[gi] = np.trace(np.matmul(block, block))
            p1s[gi] = np.trace(block)
        p2s_all.append(p2s)
        p1s_all.append(p1s)
    fig, axs = plt.subplots(5, 1)
    fulls = [sum([p2s_all[bi][gi] for bi in range(len(bs))]) for gi in range(len(gs))]
    axs[0].plot(gs, fulls)
    axs[0].set_ylabel(r'$p_2$')
    singles = [2**i for i in range(6)]
    for si in range(len(singles)):
        # axs[1].plot(gs, p2s_all[singles[si]])
        axs[1].plot(gs, p1s_all[singles[si]], color=colors[si])
        axs[1].plot(gs, p2s_all[singles[si]] / fulls, '--', color=colors[si], label='_nolegend_')
    axs[1].set_ylabel(r'$p_2(q)/p_2$')
    axs[1].legend([binary_string(singles[si], 6) for si in range(len(singles))])
    singles = [(63 ^ 2**i) for i in range(6)]
    for si in range(len(singles)):
        axs[2].plot(gs, p1s_all[singles[si]], color=colors[si])
        axs[2].plot(gs, p2s_all[singles[si]] / fulls, '--', color=colors[si], label='_nolegend_')
        # axs[2].plot(gs, p2s_all[singles[si]])
    axs[2].set_ylabel(r'$p_2(q)/p_2$')
    axs[2].legend([binary_string(singles[si], 6) for si in range(len(singles))])
    alls = [0, 2**6 - 1]
    for ai in range(len(alls)):
        axs[3].plot(gs, p1s_all[singles[ai]], color=colors[ai])
        axs[3].plot(gs, p2s_all[singles[ai]] / fulls, '--', color=colors[ai], label='_nolegend_')
        # axs[3].plot(gs, p2s_all[alls[ai]])
    axs[3].set_ylabel(r'$p_2(q)/p_2$')
    axs[3].legend([binary_string(alls[ai], 6) for ai in range(len(alls))])
    halves = [21, 42, 1+2+4]
    for hi in range(len(halves)):
        axs[4].plot(gs, p1s_all[singles[hi]], color=colors[hi])
        axs[4].plot(gs, p2s_all[singles[hi]] / fulls, '--', color=colors[hi], label='_nolegend_')
        # axs[4].plot(gs, p2s_all[halves[hi]])
    axs[4].set_ylabel(r'$p_2(q)/p_2$')
    axs[4].legend([binary_string(halves[hi], 6) for hi in range(len(halves))])
    plt.xlabel('g')
    plt.show()

b = 1

def get_Z_plaquette_projector():
    #
    #    O
    #    |
    #  __*__O
    #    |
    # bond down, bond left, in, out
    tensor_op_bottom_left = np.zeros((2, 2, 4, 4), dtype=complex)
    for i in range(2):
        tensor_op_bottom_left[0, 0, i + 2 * i, i + 2 * i] = 1
        tensor_op_bottom_left[1, 1, i + 2 * i, i + 2 * i] = 1
        tensor_op_bottom_left[0, 1, i + 2 * (i^1), i + 2 * (i^1)] = 1
        tensor_op_bottom_left[1, 0, i + 2 * (i ^ 1), i + 2 * (i ^ 1)] = 1
    # |
    # O
    #    O____
    # bond up, bond right, O top in, O right in, O top out, O right out
    tensor_op_top_right = np.zeros((2, 2, 2, 2, 2, 2), dtype=complex)
    for i in range(2):
        for j in range(2):
            tensor_op_top_right[i, j, i, j, i, j] = 1
    tensor_op_top_right = tensor_op_top_right.reshape([2, 2, 4, 4])
    op_bottom_left = tn.Node(tensor_op_bottom_left)
    op_top_right = tn.Node(tensor_op_top_right)
    op_A = bops.permute(bops.contract(op_top_right, op_bottom_left, '2', '3'), [2, 0, 1, 3, 4, 5])
    op_B = bops.permute(bops.contract(op_bottom_left, op_top_right, '2', '3'), [2, 3, 4, 0, 1, 5])
    return op_A, op_B


op_A, op_B = get_Z_plaquette_projector()
minus_boundary = np.array([0, 1], dtype=complex)
plus_boundary = np.array([1, 0], dtype=complex)
def normalize_by_gauge_invariant_projector(w, h, random_vectors, cUps, dUps, cDowns, dDowns, leftRows, rightRows):
    random_ops = [tn.Node(np.outer(vec, np.conj(vec.transpose()))) for vec in random_vectors]
    return pe.applyLocalOperators_detailedBoundary(
        cUps, dUps, cDowns, dDowns, leftRows, rightRows, op_A, op_B, h, w, random_ops)


def get_projective_boundary_operators(w, h, boundary_sectors):
    leftRows = [None for hi in range(int(h/2))]
    for hi in range(int(h/2)):
        up_vec = plus_boundary if boundary_sectors[2 * hi] == 1 else minus_boundary
        down_vec = plus_boundary if boundary_sectors[2 * hi + 1] == 1 else minus_boundary
        leftRows[hi] = tn.Node(np.outer(up_vec, down_vec).reshape([1, 2, 2, 1]))
    cUps = [None for wi in range(int(w / 2))]
    cDowns = [None for wi in range(int(w / 2))]
    dUps = [None for wi in range(int(w / 2))]
    dDowns = [None for wi in range(int(w / 2))]
    for wi in range(int(w/2)):
        cUps[wi] = tn.Node(plus_boundary.reshape([1, 2, 1])) if boundary_sectors[h+4*wi] == 1 else tn.Node(minus_boundary.reshape([1, 2, 1]))
        cDowns[wi] = tn.Node(plus_boundary.reshape([1, 2, 1])) if boundary_sectors[h+4*wi + 1] == 1 else tn.Node(minus_boundary.reshape([1, 2, 1]))
        dUps[wi] = tn.Node(plus_boundary.reshape([1, 2, 1])) if boundary_sectors[h+4*wi + 2] == 1 else tn.Node(minus_boundary.reshape([1, 2, 1]))
        dDowns[wi] = tn.Node(plus_boundary.reshape([1, 2, 1])) if boundary_sectors[h+4*wi + 3] == 1 else tn.Node(minus_boundary.reshape([1, 2, 1]))
    rightRows = [None for hi in range(h - 1)]
    rightRows[0] = \
        tn.Node(np.outer(plus_boundary, plus_boundary).reshape([1, 2, 2, 1])) \
            if boundary_sectors[h + 2 * w] == 1 else \
        tn.Node(np.outer(minus_boundary, plus_boundary).reshape([1, 2, 2, 1]))
    for hi in range(1, int(h/2)):
        up_vec = plus_boundary if boundary_sectors[h + 2 * w + 2 * hi] == 1 else minus_boundary
        down_vec = plus_boundary if boundary_sectors[h + 2 * w + 2 * hi + 1] == 1 else minus_boundary
        rightRows[hi] = tn.Node(np.outer(up_vec, down_vec).reshape([1, 2, 2, 1]))
    return cUps, dUps, cDowns, dDowns, leftRows, rightRows



# TODO start from a general basis, add optimization etc
def get_random_vectors_general_basis(w, h, cUps, dUps, cDowns, dDowns, leftRows, rightRows):
    spin_vecs = [np.array([1, 0]) * np.sqrt(2), np.array([0, 1]) * np.sqrt(2),
                 np.array([1, 1]), np.array([1, -1]), np.array([1, 1j]), np.array([1, -1j])]
    # TODO for the 1 site boundary, choose non random vectors
    random_vectors = [np.kron(spin_vecs[np.random.randint(len(spin_vecs))],
                              spin_vecs[np.random.randint(len(spin_vecs))]) for i in range(w*h)]
    norm = normalize_by_gauge_invariant_projector(w, h, random_vectors, cUps, dUps, cDowns, dDowns, leftRows, rightRows)
    if norm == 0:
        return 0
    random_vectors[0] = random_vectors[0] / np.sqrt(norm)
    return random_vectors


def get_random_operators_general_basis(w, h, n, cUps, dUps, cDowns, dDowns, leftRows, rightRows):
    random_vectors = [None for ni in range(n)]
    for ni in range(n):
        random_vectors[ni] = get_random_vectors_general_basis(w, h, cUps, dUps, cDowns, dDowns, leftRows, rightRows)
        if random_vectors[ni] == 0:
            return 0
    random_ops = [[tn.Node(np.outer(random_vectors[ni][si], np.conj(np.transpose(random_vectors[(ni + 1) % n][si]))))
                   for si in range(w * h)] for ni in range(n)]
    return random_ops


def renyi_entropy_general_basis(w, h, n, boundary_identifier):
    boundary_sectors = get_boundary_sectors(w, h, boundary_identifier)
    cUps, dUps, cDowns, dDowns, leftRows, rightRows = get_projective_boundary_operators(w, h, boundary_sectors)
    random_ops = get_random_operators_general_basis(w, h, n, cUps, dUps, cDowns, dDowns, leftRows, rightRows)
    if random_ops == 0:
        return None
    res = 1
    for ni in range(n):
        res *= pe.applyLocalOperators_detailedBoundary(
        cUps, dUps, cDowns, dDowns, leftRows, rightRows, op_A, op_B, h, w, random_ops[ni])
    return res


def double_boundary(op):
    return


def is_plus_plaquette(i):
    i_str = list(bin(i).split('b')[1])
    site_num = 6
    i_str = ['0'] * (site_num - len(i_str)) + i_str
    return np.prod([int(c) * 2 - 1 for c in i_str[1:site_num - 1]]) == 1


d = 4
full_projector_plus = \
    tn.Node(np.diag([1 if is_plus_plaquette(i) else 0 for i in range(2 ** 6)])
            .reshape([d, d, d, d, d, d]).transpose([0, 3, 1, 4, 2, 5]))
left_edge_projector_plus = tn.Node(np.diag([0, 0, 1, 1]))
left_edge_projector_minus = tn.Node(np.diag([1, 1, 0, 0]))
bottom_edge_projector_plus = tn.Node(np.diag([0, 1, 0, 1]))
bottom_edge_projector_minus = tn.Node(np.diag([1, 0, 1, 0]))
top_edge_projector_plus = tn.Node(np.diag([0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1]).reshape([4, 4, 4, 4]))
top_edge_projector_minus = tn.Node(np.diag([1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0]).reshape([4, 4, 4, 4]))
top_right_corner_projector_plus = tn.Node(np.diag([1, 0, 0, 1]))
top_right_corner_projector_minus = tn.Node(np.diag([0, 1, 1, 0]))
right_edge_projector_plus = tn.Node(np.diag([0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1]).reshape([4, 4, 4, 4]))
right_edge_projector_minus = tn.Node(np.diag([1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0]).reshape([4, 4, 4, 4]))

def exact_probability(w, h, g, d=4):
    with open('results/toricBoundaries_gauge_' + str(np.round(g, 8)), 'rb') as f:
        [upRow, downRow, leftRow, rightRow, openA, openB, A, B] = pickle.load(f)
    [cUp, dUp, te] = bops.svdTruncation(upRow, [0, 1], [2, 3], '>>')
    [cDown, dDown, te] = bops.svdTruncation(downRow, [0, 1], [2, 3], '>>')
    norm = pe.applyLocalOperators(cUp, dUp, cDown, dDown, leftRow, rightRow, A, B, w, h,
                                  [tn.Node(np.eye(4)) for i in range(w * h)])
    leftRow = bops.multNode(leftRow, 1 / norm ** (2 / w))
    num_of_boundary_options = 2**(2 * (w + h -1))
    results = np.zeros(num_of_boundary_options)
    num_of_bulk_ints = 2**((w - 1) * (h - 1))
    for boundary_i in range(num_of_boundary_options):
        list_of_sectors = get_list_of_sectors(w, h, boundary_i)
        p = 0
        for bulk_i in range(num_of_bulk_ints):
            bulk_ints = [int(c) * 2 - 1 for c in bin(bulk_i).split('b')[1]]
            bulk_ints = [-1] * ((w - 1) * (h - 1) - len(bulk_ints)) + bulk_ints
            ops = get_random_operators(w, h, 1, list_of_sectors, [bulk_ints])
            p += \
                pe.applyLocalOperators(cUp, dUp, cDown, dDown, leftRow, rightRow, A, B, h, w, [op / 4 for op in ops[0]])
        try:
            with open('results/gauge/toric_g_0.1/gauge_rep_3_g_0.1_b_' + str(boundary_i) + '_n_1_w_4_h_4_M_1000_m_999', 'rb') as f:
                curr = pickle.load(f)
                print([boundary_i, curr * 2**((w - 1) * (h - 1)) / 2**(2 * w * h)])
        except FileNotFoundError:
            pass
        results[boundary_i] = p
    return results
ps = np.zeros((2**14, len(gs)))
for gi in range(len(gs)):
    ps[:, gi] = exact_probability(4, 4, gs[gi])
    print('sum(ps) = ' + str(sum(ps[:, gi])))
with open('results/gauge/toric_ps', 'wb') as f:
    pickle.dump(ps, f)

def shrink_boundaries(upRow, downRow, leftRow, rightRow, max_bond_dim):
    M = bops.contract(upRow, rightRow, '3', '0')
    [upRow, rightRow, te] = bops.svdTruncation(M, [0, 1, 2], [3, 4, 5], '>>', maxBondDim=max_bond_dim)
    print(te)
    M = bops.contract(rightRow, downRow, '3', '0')
    [rightRow, downRow, te] = bops.svdTruncation(M, [0, 1, 2], [3, 4, 5], '>>', maxBondDim=max_bond_dim)
    print(te)
    M = bops.contract(downRow, leftRow, '3', '0')
    [downRow, leftRow, te] = bops.svdTruncation(M, [0, 1, 2], [3, 4, 5], '>>', maxBondDim=max_bond_dim)
    print(te)
    M = bops.contract(leftRow, upRow, '3', '0')
    [leftRow, upRow, te] = bops.svdTruncation(M, [0, 1, 2], [3, 4, 5], '>>', maxBondDim=max_bond_dim)
    print(te)
    return [upRow, downRow, leftRow, rightRow]


def exact_purity(w, h, g, d=4):
    with open('results/toricBoundaries_gauge_' + str(np.round(g, 8)), 'rb') as f:
        [upRow, downRow, leftRow, rightRow, openA, openB, A, B] = pickle.load(f)
        [upRow, downRow, leftRow, rightRow] = shrink_boundaries(upRow, downRow, leftRow, rightRow, max_bond_dim=4)
        [cUp, dUp, te] = bops.svdTruncation(upRow, [0, 1], [2, 3], '>>', maxTrunc=5)
        [cDown, dDown, te] = bops.svdTruncation(downRow, [0, 1], [2, 3], '>>', maxTrunc=5)
        norm = pe.applyLocalOperators(cUp, dUp, cDown, dDown, leftRow, rightRow, A, B, w, h,
                                      [tn.Node(np.eye(4)) for i in range(w * h)])
        leftRow = bops.multNode(leftRow, 1 / norm ** (2 / w))
    cUp = tn.Node(np.kron(cUp.tensor, cUp.tensor))
    dUp = tn.Node(np.kron(dUp.tensor, dUp.tensor))
    cDown = tn.Node(np.kron(cDown.tensor, cDown.tensor))
    dDown = tn.Node(np.kron(dDown.tensor, dDown.tensor))
    leftRow = tn.Node(np.kron(leftRow.tensor, leftRow.tensor))
    rightRow = tn.Node(np.kron(rightRow.tensor, rightRow.tensor))
    A = tn.Node(np.kron(A.tensor, A.tensor))
    B = tn.Node(np.kron(B.tensor, B.tensor))
    single_swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    double_swap = np.kron(single_swap, single_swap).reshape([2] * 8).transpose([3, 2, 1, 0, 4, 5, 6, 7]).reshape([2**4, 2**4])
    num_of_boundary_options = 2**(2 * (w + h -1))

    full_purity = pe.applyLocalOperators(cUp, dUp, cDown, dDown, leftRow, rightRow, A, B, h, w,
                                         [tn.Node(double_swap) for i in range(w * h)])
    return full_purity


run_estimations = False
if run_estimations:
    w = 4
    h = 4
    n = 2 
    M = 1000
    d = 4
    gs = [float(sys.argv[1])] # [0.1 * G for G in range(11)]
    num_of_boundary_options = 24 # 2**(2 * (w + h -1))
    colors = ['#0000FF', '#9D02D7', '#EA5F94', '#FA8775', '#FFB14E', '#FFD700', '#ff6f3c', '#FFD700', '#2f0056', '#930043']
    estimate_func = pe.applyLocalOperators
    for gi in range(len(gs)):
        g = gs[gi]
        dirname = sys.argv[4] + '/gauge/toric_g_' + str(g)
        try:
            os.mkdir(dirname)
        except FileExistsError:
            pass
        with open(sys.argv[4] + '/toricBoundaries_gauge_' + str(np.round(g, 8)), 'rb') as f:
            [upRow, downRow, leftRow, rightRow, openA, openB, A, B] = pickle.load(f)
        [cUp, dUp, te] = bops.svdTruncation(upRow, [0, 1], [2, 3], '>>')
        [cDown, dDown, te] = bops.svdTruncation(downRow, [0, 1], [2, 3], '>>')
        norm = pe.applyLocalOperators(cUp, dUp, cDown, dDown, leftRow, rightRow, A, B, w, h,
                                      [tn.Node(np.eye(4)) for i in range(w * h)])
        leftRow = bops.multNode(leftRow, 1 / norm ** (2 / w))

        for bi in [int(sys.argv[2])]:
            list_of_sectors = get_list_of_sectors(w, h, bi)
            ru.renyiEntropy(n, w, h, M, estimate_func, [cUp, dUp, cDown, dDown, leftRow, rightRow, A, B, w, h],
                              dirname + '/gauge_rep_' + sys.argv[3] + '_g_' + str(g) + '_b_' + str(bi),
                              get_ops_func=get_random_operators,
                              get_ops_arguments=[w, h, n, list_of_sectors])

plot_results = True
if plot_results:
    fig, axs = plt.subplots(1, 2)
    bs = [10837, 16383, 0, 2, 4, 4096, 255]
    gs = [np.round(0.1 * G, 8) for G in range(3, 11)]
    w = 4
    h = 4
    n = 2
    for bi in range(len(bs)):
        b = bs[bi]
        p2s = np.zeros(len(gs))
        vars = np.zeros(len(gs))
        for gi in range(len(gs)):
            g = gs[gi]
            with open('results/gauge/organized_toric_g_' + str(g) + '_b_' + str(b) + '_2_32', 'rb') as f:
                curr = np.array(pickle.load(f))
                curr = curr * (2**((w - 1) * (h - 1)) / 2**(2 * w * h))**n
                p2s[gi] = np.average(curr)
                vars[gi] = np.sum([np.abs(curr[i] - p2s[gi])**2 for i in range(len(curr))]) / (len(curr) - 1)
        axs[0].plot(gs, p2s)
    full_p2s = np.zeros(len(gs))
    for gi in range(len(gs)):
        g = gs[gi]
        f = exact_purity(w, h, g)
        full_p2s[gi] = f
        print(g)
    axs[1].plot(gs, full_p2s)
    plt.legend([str(b) for b in bs] + ['full'])
    plt.show()
