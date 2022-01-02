import tensornetwork as tn
import numpy as np
import pickle
import basicOperations as bops
import pepsExpect as pe
import sys

D = 2
d = 2
gs = [np.round(0.1 * i, 1) for i in range(11)]
dir = sys.argv[1]

for g in gs:
    boundary_file = dir + '/toricG/toricBoundaries_g_' + str(g)

    with open(boundary_file, 'rb') as f:
        [up_row, down_row, left_row, right_row, open_A, open_B, A, B] = pickle.load(f)
    A_double_site_open = tn.Node(bops.multiContraction(open_A, open_A, '05', '05*').tensor.\
        transpose([0, 4, 1, 5, 2, 6, 3, 7]).reshape([D**4, D**4, D**4, D**4]))
    B_double_site_open = tn.Node(bops.multiContraction(open_B, open_B, '05', '05*').tensor.\
        transpose([0, 4, 1, 5, 2, 6, 3, 7]).reshape([D**4, D**4, D**4, D**4]))
    traced_A = pe.applyOpTosite(A, tn.Node(np.eye(d)))
    A_double_site_closed = tn.Node(np.kron(traced_A.tensor, traced_A.tensor))
    traced_B = pe.applyOpTosite(B, tn.Node(np.eye(d)))
    B_double_site_closed = tn.Node(np.kron(traced_B.tensor, traced_B.tensor))
    double_up_row = tn.Node(np.kron(up_row.tensor, up_row.tensor))
    double_down_row = tn.Node(np.kron(down_row.tensor, down_row.tensor))
    double_left_row = tn.Node(np.kron(left_row.tensor, left_row.tensor))
    double_right_row = tn.Node(np.kron(right_row.tensor, right_row.tensor))

    [c_up, d_up, te] = bops.svdTruncation(double_up_row, [0, 1], [2, 3], '>>')
    [c_down, d_down, te] = bops.svdTruncation(double_down_row, [0, 1], [2, 3], '>>')


    h = 4
    left = bops.multiContraction(double_left_row, double_left_row, '3', '0')
    for i in range(int(h / 2)):
        left = bops.multiContraction(left, c_up, '5', '0', cleanOr1=True)
        left0 = B_double_site_closed
        left1 = A_double_site_open
        left2 = B_double_site_closed
        left3 = A_double_site_open
        left = bops.multiContraction(left, left0, '45', '30', cleanOr1=True, cleanOr2=True)
        left = bops.multiContraction(left, left1, '36', '30', cleanOr1=True, cleanOr2=True)
        left = bops.multiContraction(left, left2, '26', '30', cleanOr1=True, cleanOr2=True)
        left = bops.multiContraction(left, left3, '16', '30', cleanOr1=True, cleanOr2=True)
        left = bops.multiContraction(left, d_down, '06', '21', cleanOr1=True).reorder_axes([5, 4, 3, 2, 1, 0])

        left = bops.multiContraction(left, d_up, '5', '0', cleanOr1=True)
        right0 = A_double_site_open
        right1 = B_double_site_closed
        right2 = A_double_site_open
        right3 = B_double_site_closed
        left = bops.multiContraction(left, right0, '45', '30', cleanOr1=True, cleanOr2=True)
        left = bops.multiContraction(left, right1, '36', '30', cleanOr1=True, cleanOr2=True)
        left = bops.multiContraction(left, right2, '26', '30', cleanOr1=True, cleanOr2=True)
        left = bops.multiContraction(left, right3, '16', '30', cleanOr1=True, cleanOr2=True)
        left = bops.multiContraction(left, c_down, '06', '21', cleanOr1=True).reorder_axes([5, 4, 3, 2, 1, 0])
    right = bops.multiContraction(double_right_row, double_right_row, '3', '0')
    res = bops.multiContraction(left, right, '012345', '543210').tensor * 1
    with open(dir + '/toricG/checkerboard_g_' + str(g), 'wb'):
        pickle.load(res, f)

