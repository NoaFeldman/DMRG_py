import numpy as np
import tensornetwork as tn
import basicOperations as bops
import pepsExpect as pe
import PEPS as peps
import pickle
from typing import List
import scipy.linalg as linalg

# ----*-----*-----*-----*----
#   |    |     |     |     |
#   *site*     *site *     *
#   |    |     |     |     |
# ----*-----*-----*-----*----
#   |    |     |     |     |
#   *    *site *     *site *
#   |    |     |     |     |
# ----*-----*-----*-----*----

d = 2
I = np.eye(d)
X = np.array([[0, 1], [1, 0]])
Z = np.diag([1, -1])
links_in_site = 4

def plaquette_node(g=0.0) -> List[tn.Node]:
    single_site_tensor = np.zeros((d, d, d, d), dtype=complex)
    single_site_tensor[0, 0, 0, 0] = 1
    single_site_tensor[1, 1, 1, 1] = 1
    single_site = tn.Node(single_site_tensor)

    # (1+x)s that settle the plaquettes that were not chosen in the checkerboard
    # p     p
    # |     |
    # p-----p
    # physical_in, physical_out, down
    op_up_left_corner = np.zeros((d, d, d), dtype=complex)
    op_up_left_corner[:, :, 0] = I
    op_up_left_corner[:, :, 1] = X
    op_up_left = tn.Node(op_up_left_corner)
    # physical_in, physical_out, down
    projector_up_right_corner = np.zeros((d, d, d), dtype=complex)
    projector_up_right_corner[:, :, 0] = I
    projector_up_right_corner[:, :, 1] = X
    op_up_right = tn.Node(projector_up_right_corner)
    # physical_in, physical_out, up, left
    projector_down_right_corner = np.zeros((d, d, d, d), dtype=complex)
    projector_down_right_corner[:, :, 0, 0] = I
    projector_down_right_corner[:, :, 1, 1] = X
    op_down_right = tn.Node(projector_down_right_corner)
    # physical_in, physical_out, right, up
    projector_down_left_corner = np.zeros((d, d, d, d), dtype=complex)
    projector_down_left_corner[:, :, 0, 0] = I
    projector_down_left_corner[:, :, 1, 1] = X
    op_down_left = tn.Node(projector_down_left_corner)

    A = bops.unifyLegs(bops.unifyLegs(bops.unifyLegs(bops.permute(
        bops.contract(bops.contract(bops.contract(bops.contract(
        single_site, op_down_right, '0', '0'), op_down_left, '0', '0'), op_up_left, '0', '0'), op_up_right, '0', '0'),
        [1, 4, 5, 9, 7, 2, 0, 3, 6, 8]), [3, 4]), [0, 1]), [4, 5, 6, 7])
    single_pair = tn.Node(np.array([1, 0, 0, 1]).reshape([d, d]))
    left_boundary = bops.unifyLegs(bops.permute(bops.contract(bops.contract(
        single_pair, op_down_right, '0', '0'), op_up_right, '0', '0'),
        [4, 2, 1, 0, 3]), [3, 4])
    g_projector = \
        np.diag([(1 + g)**(4 - sum([int(c) for c in bin(i).split('b')[1]])) for i in range(d**4)])
    A = bops.contract(A, tn.Node(g_projector), '4', '0')
    g_pair_projector = np.diag([(1+g)**2, 1+g, 1+g, 1])
    left_boundary = bops.contract(left_boundary, tn.Node(g_pair_projector), '3', '0')

    # TODO suit BMPS for this nonequal dimension of vertical and horizontal legs
    AEnv = pe.toEnvOperator(bops.multiContraction(A, A, '4', '4*'))
    openA = tn.Node(np.kron(A.tensor, A.tensor.conj()).reshape(
        [A.tensor.shape[0]**2, A.tensor.shape[1]**2, A.tensor.shape[0]**2, A.tensor.shape[1]**2, 2**4, 2**4])\
        .transpose([4, 0, 1, 2, 3, 5]))
    upRow, downRow, leftRow, rightRow = peps.applyBMPS(AEnv, AEnv, d=(d**4))
    with open('results/toricBoundaries_squares_' + str(g), 'wb') as f:
        pickle.dump([upRow, downRow, leftRow, rightRow, openA, openA, A, A], f)
    return [A, left_boundary]

plaquette_node(g=0.1)

# |\Psi_+> from https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.119.040502
# after tracing out the ancillary right qubit.
# Fig 1 here https://www.nature.com/articles/s41534-018-0106-y.pdf, but rotated 90 degrees
#
#    *---*---*---*---*
#    | x |   | x |   | x
#    *---*---*---*---*
#  x |   | X |   | X |
#    *---*---*---*---*
#    | x |   | x |   | x
#    *---*---*---*---*
#  x |   | X |   | X |
#    *---*---*---*---*
#
# The large Xs are the nodes in my TN.
def surface_code(w, h, g=0.0):
    [A, left_boundary] = plaquette_node(g)
    # Top bounadries close the upper plaquettes.
    # bottom boundaries just require no application of additional X ops,
    # so project to 0 on the bonds.
    c_up_single = tn.Node(np.array([(1+g)**2, 0, 0, 1]).reshape([1, 4, 1]))
    c_up = tn.Node(bops.contract(c_up_single, c_up_single, '2', '0').tensor.reshape([1, 16, 1]))
    c_down_single = tn.Node(np.array([1, 0, 0, 0]).reshape([1, 4, 1]))
    c_down = tn.Node(bops.contract(c_down_single, c_down_single, '2', '0').tensor.reshape([1, 16, 1]))
    left_site = bops.unifyLegs(bops.unifyLegs(bops.unifyLegs(bops.permute(bops.contract(
        left_boundary, left_boundary, '3', '3*'), [0, 3, 1, 4, 2, 5]), [4, 5]), [2, 3]), [0, 1])

    left_row = bops.contract(left_site, left_site, '2', '0')
    # right row imposes (1+X_b) on the boundary terms.
    right_row = bops.contract(tn.Node(np.array([1, 1, 1, 1]).reshape([1, 4, 1])),
                              tn.Node(np.array([1, 1, 1, 1]).reshape([1, 4, 1])), '2', '0')
    # top left corner needs to close the plaquette
    top_left_corner = tn.Node(np.array([1, 0, 0, 1]).reshape([4, 1]))
    bottom_left_corner = tn.Node(np.array([1, 0, 0, 0]).reshape([1, 4]))
    left_rows = [bops.copyState([left_row])[0] for hi in range(int(h / 2))]
    left_rows[-1] = bops.contract(left_rows[-1], top_left_corner, '3', '0')
    left_rows[0] = bops.contract(bottom_left_corner, left_rows[0], '1', '0')
    norm = pe.applyLocalOperators_detailedBoundary(
        [c_up] * int(w / 2), [c_up] * int(w / 2), [c_down] * int(w / 2), [c_down] * int(w / 2),
        left_rows, [right_row] * int(h / 2), A, A, h, w,
        [tn.Node(np.eye(d**4)) for i in range(w * h)])
    bops.multNode(left_rows[0], 1 / norm)

    return [A, c_up, c_down, left_rows, right_row]


def get_mid_surface_explicit_block(w, h, boundary_sector=0, g=0.0, logical_qubit=0):
    [A, c_up, c_down, left_rows, right_row] = surface_code(w, h, g)
    boundary_length = 2 * w - 1 + 2 * h - 1
    boundary_indices = [int(c) * 2 - 1 for c in bin(boundary_sector).split('b')[1].zfill(boundary_length)]

    # TODO horribly inefficient, go over only allowed combinations if this code becomes important
    mid_surface_size = (2 * w - 1) * (2 * h - 1)
    allowed_combos = []
    for ind in range(2**mid_surface_size):
        combo = [int(c) * 2 - 1 for c in bin(ind).split('b')[1].zfill(mid_surface_size)]
        expanded_combo = []
        for wi in range(w - 1):
            expanded_combo = expanded_combo \
                             + combo[int((h - 0.5) * 4) * wi: int((h - 0.5) * 4) * (wi + 1)] \
                             + ['n'] * 2
        for hi in range(h - 1):
            expanded_combo += [combo[int((h - 0.5) * 4) * (w - 1) + 2 * hi]] \
                                + ['n'] * 2 \
                                + [combo[int((h - 0.5) * 4) * (w - 1) + 2 * hi + 1]]
        expanded_combo += [combo[-1]] + ['n'] * 3
        a_test_combo = [int((expanded_combo[i] + 1) / 2) if expanded_combo[i] == -1 or expanded_combo[i] == 1
                      else expanded_combo[i]
                      for i in range(len(expanded_combo))]
        legal = True
        boundary_counter = 0
        # left boundaries
        for hi in range(h - 1):
            if expanded_combo[hi * 4] * expanded_combo[hi * 4 + 3] != boundary_indices[boundary_counter]:
                legal = False
                break
            boundary_counter += 1
        if not legal:
            continue
        if expanded_combo[(h - 1) * 4] != boundary_indices[boundary_counter]:
            continue
        boundary_counter += 1
        # top bottom boundaries
        for wi in range(w - 1):
            if expanded_combo[wi * h * 4] * expanded_combo[wi * h * 4 + 1] \
                != boundary_indices[boundary_counter]:
                legal = False
                break
            boundary_counter += 1
            if expanded_combo[wi * 4 * h + int((h - 0.5) * 4) - 1] * expanded_combo[(wi + 1) * 4 * h + (h - 1) * 4] \
                != boundary_indices[boundary_counter]:
                legal = False
                break
            boundary_counter += 1
        if not legal:
            continue
        # right boundaries
        if expanded_combo[4 * h * (w - 1)] != boundary_indices[boundary_counter]:
            continue
        boundary_counter += 1
        for hi in range(h - 1):
            if expanded_combo[4 * h * (w - 1) + 4 * hi + 3] * expanded_combo[4 * h * (w - 1) + 4 * hi + 4] \
                != boundary_indices[boundary_counter]:
                legal = False
            boundary_counter += 1
        if not legal:
            continue
        # mid stars
        for wi in range(w - 1):
            for hi in range(h - 1):
                if np.prod(expanded_combo[4 * h * wi + 4 * hi + 2: 4 * h * wi + 4 * (hi + 1) + 2]) != 1:
                    legal = False
                    break
            for hi in range(h - 1):
                if expanded_combo[4 * h * wi + 4 * hi + 2] * \
                        expanded_combo[4 * h * wi + 4 * hi + 5] * \
                        expanded_combo[4 * h * wi + 4 * hi + 2 + 5 + 4 * (h - 1)] * \
                        expanded_combo[4 * h * wi + 4 * hi + 2 + 6 + 4 * (h - 1)] != 1:
                    legal = False
                    break
        if not legal:
            continue
        allowed_combos.append(expanded_combo)

    allowed_combos_temp = []
    for c in allowed_combos:
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    temp = c.copy()
                    temp[9] = i
                    temp[10] = j
                    temp[14] = k
                    allowed_combos_temp.append(temp)
    allowed_combos = allowed_combos_temp
    #                    00                           01                          10                 11
    projectors_ops = [np.diag([1, 0]), np.array([[0 , 1], [0, 0]]), np.array([[0, 0], [1, 0]]), np.diag([0, 1])]
    block = np.zeros((len(allowed_combos), len(allowed_combos)))
    for ci in range(len(allowed_combos)):
        for cj in range(len(allowed_combos)):
            ops = []
            for ni in range(int(len(allowed_combos[ci]) / 4)):
                op = np.eye(1)
                for si in range(4):
                    curr_op = np.eye(d) if allowed_combos[ci][ni * 4 + si] == 'n' else \
                        projectors_ops[int((allowed_combos[ci][ni * 4 + si] + 1) / 2
                                           + 2 * (allowed_combos[cj][ni * 4 + si] + 1) / 2)]
                    op = np.kron(op, curr_op)
                ops.append(op)
            if logical_qubit == 1 & w == 2 and h == 2: # TODO sort for other system measures
                ops[1] = np.matmul(np.kron(X, np.kron(X, np.kron(I, I))),
                                   np.matmul(ops[1], np.kron(X, np.kron(X, np.kron(I, I)))))
                ops[3] = np.matmul(np.kron(X, np.kron(X, np.kron(I, I))),
                                   np.matmul(ops[1], np.kron(X, np.kron(X, np.kron(I, I)))))
            block[ci, cj] = pe.applyLocalOperators_detailedBoundary(
                [c_up] * int(w / 2), [c_up] * int(w / 2), [c_down] * int(w / 2), [c_down] * int(w / 2),
                left_rows, [right_row] * int(h / 2), A, A, h, w, [tn.Node(op) for op in ops])
    b = 1
    return block


dt = 1e-2
int_op_top_left_tensor = np.zeros((d**4, d**4, 2))
int_op_top_left_tensor[:, :, 0] = np.eye(d**4)
int_op_top_left_tensor[:, :, 1] = np.kron(I, np.kron(linalg.expm(dt * X), np.kron(I, I)))
int_op_top_left = tn.Node(int_op_top_left_tensor)
int_op_top_right_tensor = np.zeros((d**4, d**4, 2, 2))
int_op_top_right_tensor[:, :, 0, 0] = np.eye(d ** 4)
int_op_top_right_tensor[:, :, 1, 1] = np.kron(I, np.kron(I, np.kron(linalg.expm(dt * X), I)))
int_op_top_right = tn.Node(int_op_top_right_tensor)
int_op_bottom_right_tensor = np.zeros((d ** 4, d ** 4, 2, 2))
int_op_bottom_right_tensor[:, :, 0, 0] = np.eye(d ** 4)
int_op_bottom_right_tensor[:, :, 1, 1] = np.kron(I, np.kron(I, np.kron(I, linalg.expm(dt * X))))
int_op_bottom_right = tn.Node(int_op_bottom_right_tensor)
int_op_bottom_left_tensor = np.zeros((d ** 4, d ** 4, 2))
int_op_bottom_left_tensor[:, :, 0] = np.eye(d ** 4)
int_op_bottom_left_tensor[:, :, 1] = np.kron(linalg.expm(dt * X), np.kron(I, np.kron(I, I)))
int_op_bottom_left = tn.Node(int_op_bottom_left_tensor)
def apply_interaction_op(A, B, C, D, bond_dim_vertical, bond_dim_horizontal):
    AB = bops.contract(bops.contract(
        A, int_op_top_left, '4', '1'), bops.contract(B, int_op_top_right, '4', '1'), '15', '35')
    [A, B, te] = bops.svdTruncation(AB, [0, 1, 2, 3], [4, 5, 6, 7, 8], '<<', maxBondDim=bond_dim_horizontal, normalize=True)
    if len(te) > 0:
        print(max(te))
    A = bops.permute(A, [0, 4, 1, 2, 3])
    BC = bops.contract(B, bops.contract(C, int_op_bottom_right, '4', '1'), '35', '05')
    [B, C, te] = bops.svdTruncation(BC, [0, 1, 2, 3], [4, 5, 6, 7, 8], '>>', maxBondDim=bond_dim_vertical, normalize=True)
    if len(te) > 0:
        print(max(te))
    B = bops.permute(B, [1, 2, 4, 0, 3])
    CD = bops.contract(C, bops.contract(D, int_op_bottom_left, '4', '1'), '35', '15')
    [C, D, te] = bops.svdTruncation(CD, [0, 1, 2, 3], [4, 5, 6, 7], '>>', maxBondDim=bond_dim_horizontal, normalize=True)
    if len(te) > 0:
        print(max(te))
    C = bops.permute(C, [0, 1, 2, 4, 3])
    D = bops.permute(D, [1, 0, 2, 3, 4])
    return A, B, C, D

# playing around with the pure gauge model of \sum_p X_p + g * \sum_l Z_l
def gauge_models_test(g, A=None, B=None, C=None, D=None):
    if A == None:
        [A, left_boundary] = plaquette_node()
        A = tn.Node(np.real(A.tensor))
        B = tn.Node(np.copy(A.tensor))
        C = tn.Node(np.copy(A.tensor))
        D = tn.Node(np.copy(A.tensor))
    local_op_X = np.kron(X, np.kron(X, np.kron(X, X))).reshape([d**links_in_site, d**links_in_site])
    local_op_Z = np.zeros((d**links_in_site, d**links_in_site))
    for i in range(links_in_site):
        local_op_Z += np.kron(np.eye(d**i), np.kron(g * Z, np.eye(d**(links_in_site - 1 - i))))
    local_op_Z = local_op_Z.reshape([d**links_in_site, d**links_in_site])
    local_op = local_op_X + local_op_Z
    local_op_exp = tn.Node(linalg.expm(dt * local_op))

    num_of_steps = int(10 / dt)
    bond_dim_vertical = 8
    bond_dim_horizontal = 4
    for stepi in range(num_of_steps):
        A = bops.contract(A, local_op_exp, '4', '1')
        B = bops.contract(B, local_op_exp, '4', '1')
        C = bops.contract(C, local_op_exp, '4', '1')
        D = bops.contract(D, local_op_exp, '4', '1')
        A, B, C, D = apply_interaction_op(A, B, C, D, bond_dim_vertical, bond_dim_horizontal)
        A, B, C, D = apply_interaction_op(B, A, D, C, bond_dim_vertical, bond_dim_horizontal)
        A, B, C, D = apply_interaction_op(D, C, B, A, bond_dim_vertical, bond_dim_horizontal)
        A, B, C, D = apply_interaction_op(C, D, A, B, bond_dim_vertical, bond_dim_horizontal)

    with open('results/gauge/imag_time_evolution_g_' + str(g), 'wb') as f:
        pickle.dump([A, B, C, D], f)

gs = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0]
Z_s_vertical = tn.Node(np.kron(Z, np.kron(Z, np.kron(I, np.kron(I, np.kron(I, np.kron(I, np.kron(Z, Z))))))).reshape([d**4] * 4))
Z_s_horizontal = tn.Node(np.kron(Z, np.kron(I, np.kron(I, np.kron(Z, np.kron(I, np.kron(Z, np.kron(Z, I))))))).reshape([d**4] * 4))
for g in gs:
    print('-----')
    print('g = ' + str(g))
    try:
        with open('results/gauge/imag_time_evolution_g_' + str(g), 'rb') as f:
            A = pickle.load(f)
    except FileNotFoundError:
        plaquette_node(g)
