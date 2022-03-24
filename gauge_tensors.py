import numpy as np
import tensornetwork as tn
import basicOperations as bops
import pepsExpect as pe
import PEPS as peps


def toric_code():
    #  |
    #  O
    #  |
    #  *__O__
    # bond up, bond right, * in, O up in, O down in, * out, O up out, O down out
    tensor_op_bottom_left = np.zeros((2, 2, 1, 2, 2, 1, 2, 2), dtype=complex)
    for i in range(2):
        for j in range(2):
            # identity
            tensor_op_bottom_left[0, 0, 0, i, j, 0, i, j] = 1
            # X
            tensor_op_bottom_left[1, 1, 0, i, j, 0, i ^ 1, j ^ 1] = 1
    #  --O
    #  |
    # bond down, O in, O out
    tensor_op_top = np.zeros((2, 2, 2), dtype=complex)
    for i in range(2):
        tensor_op_top[0, i, i] = 1
        tensor_op_top[1, i, i ^ 1] = 1
    #   O
    # __|
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

    A = bops.permute(bops.contract(bops.contract(bops.contract(
        base_site, op_bottom_left, '012', '234'), op_top, '3', '1'), op_right, '3', '1'), [1, 0, 3, 5, 2, 6, 4])
    B = bops.permute(bops.contract(bops.contract(bops.contract(
        base_site, op_top, '2', '1'), op_right, '1', '1'), op_bottom_left, '024', '234'), [3, 2, 0, 1, 4, 6, 5])

    A = tn.Node(A.tensor.reshape([2, 2, 2, 2, 4]))
    B = tn.Node(B.tensor.reshape([2, 2, 2, 2, 4]))
    AEnv = pe.toEnvOperator(bops.multiContraction(A, A, '4', '4*'))
    BEnv = pe.toEnvOperator(bops.multiContraction(B, B, '4', '4*'))
    upRow, downRow, leftRow, rightRow, openA, openB = peps.applyBMPS(A, B, d=4)
    circle = bops.contract(bops.contract(bops.contract(
        upRow, rightRow, '3', '0'), downRow, '5', '0'), leftRow, '70', '03')
    BA = bops.contract(openB, openA, '2', '4')
    AB = bops.contract(openA, openB, '2', '4')

    b = 1

toric_code()