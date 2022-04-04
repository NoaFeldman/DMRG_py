import numpy as np
import tensornetwork as tn
import basicOperations as bops
import pepsExpect as pe
import PEPS as peps
import pickle
import randomUs as ru
from typing import List


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
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    Hs_left = np.eye(1)
    Hs_right = np.eye(1)
    for i in range(8):
        Hs_left = np.kron(Hs_left, np.array([[1, 1/(1+g)], [1, -1/(1+g)]]))
        Hs_right = np.kron(Hs_right, np.array([[1, 1+g], [1, -1-g]]))
    rotated_mat = np.round(np.matmul(Hs_left, np.matmul(mat, Hs_right)), 8)
    id = tn.Node(np.eye(4))
    traced_out = bops.contract(bops.contract(bops.contract(M, id, '67', '01'), id, '45', '01'), id, '23', '01')
    b = 1

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
    return int_results


up_spin = np.array([1, 0])
down_spin = np.array([0, 1])
def get_random_local_vectors(w, h, list_of_sectors) -> List[np.array]:
    int_results = get_local_vectors(w, h, list_of_sectors, [np.random.randint(2) * 2 - 1 for i in range((h-1) * (w - 1))])
    if int_results is None:
        return None
    results = [None for i in range(w * h)]
    for wi in range(w):
        for hi in range(h):
            top_vec = up_spin if int_results[2 * wi, hi] == 1 else down_spin
            right_vec = up_spin if int_results[2 * wi + 1, hi] == 1 else down_spin
            curr = np.kron(top_vec, right_vec)
            results[hi + wi * h] = curr
    return results


# boundary_sectors are the sector eigenvalues on the boundary from the top left corner and down the left edge,
# then top-bottom for the middle, and then top to bottom in the right edge.
def get_list_of_sectors(w, h, boundary_identifier):
    # After all of the boundary sections but one are set, the last one is determined by them
    boundary_sectors = [int(c) * 2 - 1 for c in bin(boundary_identifier).split('b')[1]]
    boundary_sectors = [-1] * (boundary_length - len(boundary_sectors)) + boundary_sectors
    boundary_sectors.append(np.prod(boundary_sectors))
    res = [[boundary_sectors[i] for i in range(h)]]
    for i in range(1, w):
        res.append([boundary_sectors[h - 2 + i * 2]] + [1] * (h - 1) + [boundary_sectors[h - 1 + i * 2]])
    res.append([boundary_sectors[i] for i in range(-(h+1), 0)])
    return res




def get_random_operators(w, h, n, list_of_sectors):
    random_local_vectors = [None for ni in range(n)]
    for ni in range(n):
        random_local_vectors[ni] = get_random_local_vectors(w, h, list_of_sectors)
        if random_local_vectors[ni] is None:
            return None
    results = [None for ni in range(n)]
    for ni in range(n):
        results[ni] = [tn.Node(np.outer(random_local_vectors[ni][si], random_local_vectors[(ni + 1) % n][si]))
                       for si in range(w * h)]
    return results


def get_explicit_block(w, h):
    res = []
    num_of_free_site_choices = (h - 1) * (w - 1)
    list_of_sectors = get_list_of_sectors(w, h, [1]  * (h * 2 + 1 + 2 * (w - 1)))
    for i in range(2**num_of_free_site_choices):
        choices = [int(c) * 2 - 1 for c in bin(i).split('b')[1]]
        choices = [-1] * (num_of_free_site_choices - len(choices)) + choices
        local_vectors = get_local_vectors(w, h, list_of_sectors, choices)
        global_vector = np.eye(1)
        for hi in range(h):
            for wi in range(w):
                global_vector = np.kron(global_vector, local_vectors[hi, wi])
        res.append(global_vector)
    b = 1

w = 2
h = 2
n = 2
M = 1000
d = 4

estimate_func = pe.applyLocalOperators
with open('results/toricBoundaries_gauge_', 'rb') as f:
    [upRow, downRow, leftRow, rightRow, openA, openB, A, B] = pickle.load(f)
[cUp, dUp, te] = bops.svdTruncation(upRow, [0, 1], [2, 3], '>>')
[cDown, dDown, te] = bops.svdTruncation(downRow, [0, 1], [2, 3], '>>')
norm = pe.applyLocalOperators(cUp, dUp, cDown, dDown, leftRow, rightRow, A, B, w, h,
                                  [tn.Node(np.eye(d)) for i in range(w * h)])
leftRow = bops.multNode(leftRow, 1 / norm ** (2 / w))
circle = bops.contract(bops.contract(bops.contract(
    upRow, rightRow, '3', '0'), downRow, '5', '0'), leftRow, '70', '03')

M = bops.contract(bops.contract(bops.contract(bops.contract(
    circle, openB, '07', '14'), openA, '017', '124'), openA, '523', '134'), openB, '5018', '1234')
mat = np.real(M.tensor.transpose([0, 4, 2, 6, 1, 5, 3, 7]). reshape([4**4, 4**4]))
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
Hs = np.eye(1)
for i in range(8):
    Hs = np.kron(Hs, H)
rotated_mat = np.matmul(Hs, np.matmul(mat, Hs))
rotated_mat = np.round(rotated_mat / np.trace(rotated_mat), 8)
boundary_length = h * 2 + 1 + 2 * (w - 1) - 1
steps_num = 1000
for si in range(2**(boundary_length)):
    list_of_sectors = get_list_of_sectors(w, h, si)
    mysum = 0
    for i in range(steps_num):
        random_ops = get_random_operators(w, h, n, list_of_sectors)
        if random_ops == None:
            break
        tst0 = np.kron(random_ops[0][0].tensor, np.kron(random_ops[0][1].tensor, np.kron(random_ops[0][2].tensor, random_ops[0][3].tensor)))
        tst1 = np.kron(random_ops[1][0].tensor, np.kron(random_ops[1][1].tensor, np.kron(random_ops[1][2].tensor, random_ops[1][3].tensor)))
        exact_result = np.round(np.trace(np.matmul(mat, tst0)) * np.trace(np.matmul(mat, tst1)), 8)
        peps_result = pe.applyLocalOperators(cUp, dUp, cDown, dDown, leftRow, rightRow, A, B, w, h, random_ops[0]) * \
            pe.applyLocalOperators(cUp, dUp, cDown, dDown, leftRow, rightRow, A, B, w, h, random_ops[1])
        peps_result = np.round(peps_result, 8)
        if peps_result != exact_result:
            print(si, exact_result, peps_result)
        mysum += peps_result
    block_size = 2**((w - 1) * (h - 1))
    hilbert_space_size = 2**(2 * w * 2 * h)
    norm = block_size / hilbert_space_size
    avg = mysum / steps_num / norm**n
    print(list_of_sectors, np.log(avg) / np.log(2))
    b = 1

# ru.renyiEntropy(n, w, h, M, estimate_func, [cUp, dUp, cDown, dDown, leftRow, rightRow, A, B, w, h],
#                       'results/gauge/gauge_rep_' + str(1), get_ops_func=get_random_operators,
#                       get_ops_arguments=[w, h, n, list_of_sectors])