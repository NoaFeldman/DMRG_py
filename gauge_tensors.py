import numpy as np
import tensornetwork as tn
import basicOperations as bops
import pepsExpect as pe
import PEPS as peps
import pickle


def toric_code():
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
    # bond left, ) in, O out
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

    A = tn.Node(A.tensor.reshape([2, 2, 2, 2, 4]))
    B = tn.Node(B.tensor.reshape([2, 2, 2, 2, 4]))
    AEnv = pe.toEnvOperator(bops.multiContraction(A, A, '4', '4*'))
    BEnv = pe.toEnvOperator(bops.multiContraction(B, B, '4', '4*'))
    upRow, downRow, leftRow, rightRow, openA, openB = peps.applyBMPS(A, B, d=4)
    with open('results/toricBoundaries_gauge_', 'wb') as f:
        pickle.dump([upRow, downRow, leftRow, rightRow, openA, openB, A, B], f)
    circle = bops.contract(bops.contract(bops.contract(
        upRow, rightRow, '3', '0'), downRow, '5', '0'), leftRow, '70', '03')
    M = bops.contract(bops.contract(bops.contract(bops.contract(
        circle, openB, '07', '14'), openA, '016', '124'), openA, '423', '134'), openB, '2013', '1234')
    mat = np.real(M.tensor.transpose([0, 2, 6, 4, 1, 3, 7, 5]). reshape([4**4, 4**4]) / 0.03125)
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    Hs = np.eye(1)
    for i in range(8):
        Hs = np.kron(Hs, H)
    rotated_mat = np.matmul(Hs, np.matmul(mat, Hs))
    bulkX = np.eye(2)
    for i in range(3):
        bulkX = np.kron(bulkX, np.array([[0, 1], [1, 0]]))
    bulkX = np.kron(bulkX, np.eye(4))
    bulkX = np.kron(bulkX, np.array([[0, 1], [1, 0]]))
    bulkX = np.kron(bulkX, np.eye(2))
    id = tn.Node(np.eye(4))
    traced_out = bops.contract(bops.contract(bops.contract(M, id, '67', '01'), id, '45', '01'), id, '23', '01')
    b = 1



pls = np.array([1, 1]) / np.sqrt(2)
mns = np.array([1, -1]) / np.sqrt(2)

# w and h are the dimensions of the system *tensor-wise* and not site-wise (when each tensor is two sites).
# list_of_sectors is a list of lists, from left to write and top to bottom.
# For a w*h system, the first w lists are of length 1+h (including edges).
# The last list is of length h, only right edges.
def random_local_vectors(w, h, list_of_sectors):
    int_results = np.zeros((2 * w, h))

    # left edge sectors
    int_results[0, 0] = 1 if list_of_sectors[0][0] == 1 else -1
    for ri in range(h - 1):
        int_results[1, ri] = np.random.randint(2) * 2 - 1
        int_results[0, ri + 1] = list_of_sectors[0][ri + 1] / (int_results[0, ri] * int_results[1, ri])
    int_results[1, h - 1] = list_of_sectors[0][h] / int_results[0, h - 1]

    # bulk columns
    for ci in range(1, w):
        int_results[2 * ci, 0] = 1 if list_of_sectors[ci][0] == 1 else -1
        for ri in range(h - 1):
            if ci == w - 1:
                int_results[2 * ci + 1, ri] = list_of_sectors[w][ri]
            else:
                int_results[2 * ci + 1, ri] = np.random.randint(2) * 2 - 1
            int_results[2 * ci, ri + 1] = list_of_sectors[ci][ri + 1] / \
                                          (int_results[2 * ci - 1, ri] * int_results[2 * ci, ri] * int_results[2 * ci + 1, ri])
        int_results[2 * ci + 1, h - 1] = list_of_sectors[ci][h] / (int_results[2 * ci - 1, h - 1] * int_results[2 * ci, h - 1])
    # if necessary, fix bottom right corner
    if int_results[2 * w - 1, h - 1] != list_of_sectors[w][h - 1]:
        int_results[2 * w - 1, h - 1] *= -1
        int_results[2 * w - 3, h - 1] *= -1
        int_results[2 * w - 4, h - 1] *= -1
        int_results[2 * w - 3, h - 2] *= -1

    results = [None for i in range(w * h)]
    for wi in range(w):
        for hi in range(h):
            top_vec = pls if int_results[2 * wi, hi] == 1 else mns
            right_vec = pls if int_results[2 * wi + 1, hi] == 1 else mns
            curr = np.kron(top_vec, right_vec)
            results[wi * h + hi] = tn.Node(curr)
    # TODO check new AB BMPS
    # TODO check everything for w = 2, h = 1
    return results

toric_code()
w = 3
h = 4
list_of_sectors = [[np.random.randint(2) * 2 - 1 for i in range(h + 1)] for j in range(w + 1)]
results = random_local_vectors(w, h, list_of_sectors)
b = 1