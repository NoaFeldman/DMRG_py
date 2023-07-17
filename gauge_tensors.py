import numpy as np
import tensornetwork as tn
import basicOperations as bops
import pepsExpect as pe
import PEPS as peps
import pickle
import os
import sys
import gc
import tdvp
import corner_tm as ctm
import scipy.sparse as sparse

X = np.array([[0, 1], [1, 0]])
I = np.eye(2)
d = 2
alpha = 0
beta = 0
gamma = 0
delta = 0


# |\psi> = \prod_p (1 + cX^p)|0>
# This construction does not obey Erez's requirement, but it has D=2
def get_toric_c_tensors(c):
    A_tensor = np.zeros((2, 2, 2, 2, 2, 2), dtype=complex)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                A_tensor[i, i, k, j, (i + j) % 2, (i + k) % 2] = c ** i
    A = tn.Node(A_tensor.reshape([2] * 4 + [4]))
    boundaries_filename = 'results/gauge/toric_c/toricBoundaries_c_' + str(c)
    if not os.path.exists(boundaries_filename):
        AEnv = tn.Node(bops.permute(bops.contract(A, A, '4', '4*'), [0, 4, 1, 5, 2, 6, 3, 7]).tensor.reshape([4] * 4))
        upRow, downRow, leftRow, rightRow = peps.applyBMPS(AEnv, AEnv, d=d ** 2)
        openA = tn.Node(np.kron(A.tensor, A.tensor.conj()).reshape([4] * 4 + [4, 4]).transpose([4, 0, 1, 2, 3, 5]))
        with open(boundaries_filename, 'wb') as f:
            pickle.dump([upRow, downRow, leftRow, rightRow, openA, openA, A, A], f)
    [cUp, dUp, cDown, dDown, leftRow, rightRow, openA, openB, A, B] = \
        get_boundaries_from_file(boundaries_filename, 2, 2)
    return [cUp, dUp, cDown, dDown, leftRow, rightRow, openA, openB, A, B]


def square_wilson_loop_expectation_value(cUp: tn.Node, dUp: tn.Node, cDown: tn.Node, dDown: tn.Node,
                                         leftRow: tn.Node, rightRow: tn.Node, openA: tn.Node, L: int, chi=128):
    tau_projector = tn.Node(np.eye(openA[1].dimension))
    X = np.array([[0, 1], [1, 0]])
    I = np.eye(2)
    wilson = large_system_expectation_value(L, L, cUp, dUp, cDown, dDown, leftRow, rightRow, openA,
            tn.Node(np.eye(openA[1].dimension)),
            [[tn.Node(np.kron(I, X))] * (L - 1) + [tn.Node(np.kron(I, I))]] + \
            [[tn.Node(np.kron(X, I))] + [tn.Node(np.kron(I, I))] * (L - 2) + [tn.Node(np.kron(X, I))]] * (L - 2) + \
            [[tn.Node(np.kron(X, X))] + [tn.Node(np.kron(I, X))] * (L - 2) + [tn.Node(np.kron(X, I))]], chi=chi)
    norm = large_system_expectation_value(L, L, cUp, dUp, cDown, dDown, leftRow, rightRow, openA, tau_projector,
                                          [[tn.Node(np.eye(openA[0].dimension))] * L] * L, chi=chi)
    if norm == 0:
        large_system_expectation_value(L, L, cUp, dUp, cDown, dDown, leftRow, rightRow, openA, tau_projector,
                                       [[tn.Node(np.eye(openA[0].dimension))] * L] * L, chi=chi)
    print(L, wilson, norm, wilson / norm)
    return wilson / norm


def get_boundaries_from_file(filename, w, h):
    with open(filename, 'rb') as f:
        [upRow, downRow, leftRow, rightRow, openA, openB, A, B] = pickle.load(f)
        [upRow, downRow, leftRow, rightRow, te] = shrink_boundaries(upRow, downRow, leftRow, rightRow, bond_dim=2)
        [cUp, dUp, te] = bops.svdTruncation(upRow, [0, 1], [2, 3], '>>', maxTrunc=5)
        [cDown, dDown, te] = bops.svdTruncation(downRow, [0, 1], [2, 3], '>>', maxTrunc=5)
        norm = pe.applyLocalOperators(cUp, dUp, cDown, dDown, leftRow, rightRow, A, B, h, w,
                                      [tn.Node(np.eye(4)) for i in range(w * h)])
        leftRow = bops.multNode(leftRow, 1 / norm ** (2 / h))
        [cUp, dUp, te] = bops.svdTruncation(upRow, [0, 1], [2, 3], '>>', maxBondDim=upRow.tensor.shape[0])
        [cDown, dDown, te] = bops.svdTruncation(downRow, [0, 1], [2, 3], '>>', maxBondDim=downRow.tensor.shape[0])
    return [cUp, dUp, cDown, dDown, leftRow, rightRow, openA, openB, A, B]


def get_choice_indices(boundary, choice, n, d=2):
    result = [boundary[0], choice[0],
              (boundary[0] + boundary[1] + choice[0]) % d,
              (boundary[0] + boundary[1] + boundary[2] + choice[0]) % d]
    for ni in range(1, n - 1):
        result += [boundary[ni * 2 + 1], choice[ni],
                   (choice[ni - 1] + choice[ni] + boundary[ni * 2 + 1]) % d,
                   (np.sum(boundary[:(2 * ni + 3)]) + choice[ni]) % d]
    result += [boundary[n * 2 - 1], boundary[-1],
               (boundary[-1] + choice[-1] + boundary[n * 2 - 1]) % d, np.sum(boundary) % d]
    return result


# TODO here only d = 2
def get_2_by_n_explicit_block(filename, n, bi, d=2):
    [cUp, dUp, cDown, dDown, leftRow, rightRow, openA, openB, A, B] = get_boundaries_from_file(filename, w=n, h=2)
    boundary = [int(c) for c in bin(bi).split('b')[1].zfill(2 * n + 2)]
    projectors = [[np.diag([1, 0]), np.array([[0, 0], [1, 0]])],
                  [np.array([[0, 1], [0, 0]]), np.diag([0, 1])]]

    num_of_choices = n - 1
    block = np.zeros((d ** num_of_choices, d ** num_of_choices), dtype=complex)
    choices = [[int(c) for c in bin(choice).split('b')[1].zfill(num_of_choices)] for choice in
               range(d ** num_of_choices)]
    for ci in range(len(choices)):
        ingoing = get_choice_indices(boundary, choices[ci], n)
        for cj in range(len(choices)):
            outgoing = get_choice_indices(boundary, choices[cj], n)
            ops = [tn.Node(np.kron(projectors[ingoing[2 * i]][outgoing[2 * i]],
                                   projectors[ingoing[2 * i + 1]][outgoing[2 * i + 1]])) for i in
                   range(int(len(ingoing) / 2))]
            block[ci, cj] = pe.applyLocalOperators(cUp, dUp, cDown, dDown, leftRow, rightRow, openA, openB, h=2, w=n,
                                                   ops=ops)
    return block


def get_zohar_tensor(alpha, betas, gamma, delta):
    tensor = np.zeros([d] * 6, dtype=complex)
    tensor[0, 0, 0, 0, 0, 0] = alpha
    tensor[1, 1, 0, 0, 1, 1] = betas[0]
    tensor[1, 0, 0, 1, 1, 0] = betas[1]
    tensor[0, 0, 1, 1, 0, 0] = betas[2]
    tensor[0, 1, 1, 0, 0, 1] = betas[3]
    tensor[1, 0, 1, 0, 1, 0] = gamma
    tensor[0, 1, 0, 1, 0, 1] = gamma
    tensor[1, 1, 1, 1, 1, 1] = delta
    return tensor


def toric_tensors_lgt_approach(model, param, d=2):
    global alpha
    global beta
    global gamma
    global delta
    # u, r, d, l, t, s
    tensor = np.zeros([d] * 6, dtype=complex)
    for i in range(d):
        for j in range(d):
            for k in range(d):
                tensor[i, j, k,
                int(np.round(np.angle(np.exp(2j * np.pi * i / d) *
                                      np.exp(2j * np.pi * j / d) *
                                      np.exp(-2j * np.pi * k / d)) * d / (2 * np.pi), 10)) % d,
                i, j] = 1
    if model == 'zeros_diff':
        tensor[0, 0, 0, 0, 0, 0] = param
    elif model == 'zohar_alpha':
        tensor[0, 0, 0, 0, 0, 0] = param
    elif model[:10] == 'vary_alpha':
        alpha = param
        tensor = get_zohar_tensor(alpha, beta * np.ones(4), gamma, delta)
    elif model[:9] == 'vary_beta':
        beta = param
        tensor = get_zohar_tensor(alpha, beta * np.ones(4), gamma, delta)
    elif model[:10] == 'vary_gamma':
        gamma = param
        tensor = get_zohar_tensor(alpha, beta * np.ones(4), gamma, delta)
    elif model[:7] == 'vary_ad':
        alpha = param
        delta = param
        tensor = get_zohar_tensor(alpha, beta * np.ones(4), gamma, delta)
    elif model[:14] == 'zohar_non_symm':
        gamma = param
        tensor = get_zohar_tensor(alpha, [beta_ur, beta_lu, beta_dl, beta_rd], gamma, delta)
    elif model == 'zohar_gamma':
        tensor[1, 0, 1, 0, 1, 0] = param
        tensor[0, 1, 0, 1, 0, 1] = param
    elif model == 'orus':
        for i in range(d):
            for j in range(d):
                tensor[i, j, :, :, :, :] *= (1 + param) ** (i + j)
        tensor /= ((1 + param) ** 2)**0.75
    elif model == 'zohar':
        alpha, beta, gamma, delta = param
        tensor[0, 0, 0, 0, 0, 0] = alpha
        for i in range(2):
            for j in range(2):
                tensor[i, j, (i + 1) % 2, (j + 1) % 2, i, j] = beta
        tensor[0, 1, 0, 1, 0, 1] = gamma
        tensor[1, 0, 1, 0, 1, 0] = gamma
        tensor[1, 1, 1, 1, 1, 1] = delta
        tensor[1, 1, 1, 1, 1, 1] = delta
    elif model == 'toric_c':
        tensor = np.zeros([2] * 6, dtype=complex)
        tensor[0, 0, 0, 0, 0, 0] = 1
        tensor[1, 1, 1, 1, 0, 0] = param
        tensor[0, 0, 1, 0, 0, 1] = param**0.25
        tensor[1, 1, 0, 1, 0, 1] = param**0.75
        tensor[0, 0, 0, 1, 1, 0] = param**0.25
        tensor[1, 1, 1, 0, 1, 0] = param**0.75
        tensor[0, 0, 1, 1, 1, 1] = param**0.5
        tensor[1, 1, 0, 0, 1, 1] = param**0.5
        # tensor /= param
    elif model == 'toric_c_constructed':
        tensor = np.zeros([2] * 10, dtype=complex)
        tensor[0, 0, 0, 0, 0, 0, 0, 0, 0, 0] = 1
        A = tn.Node(tensor)
        x = tn.Node(X)
        A.tensor = A.tensor + param ** 0.25 * bops.contract(bops.contract(bops.permute(bops.contract(bops.contract(
            x, A, '1', '0'), x, '2', '1'), [0, 1, 9] + list(range(2, 9))), x, '8', '1'), x, '8', '1').tensor
        A.tensor = A.tensor + param ** 0.25 * bops.contract(bops.permute(bops.contract(bops.permute(bops.contract(
            A, x, '3', '1'), [0, 1, 2, 9] + list(range(3, 9))), x, '4', '1'), [0, 1, 2, 3, 9] + list(range(4, 9))),
            x, '9', '1').tensor
        A.tensor = A.tensor + param ** 0.25 * bops.permute(bops.contract(bops.permute(bops.contract(
            A, x, '5', '1'), list(range(5)) + [9, 5, 6, 7, 8]), x, '7', '1'), list(range(7)) + [9, 7, 8]).tensor
        A.tensor = A.tensor + param ** 0.25 * bops.permute(
            bops.contract(bops.permute(bops.contract(bops.permute(bops.contract(
                A, x, '1', '1'), [0, 9] + list(range(1, 9))), x, '7', '1'), list(range(7)) + [9, 7, 8]),
                x, '8', '1'), list(range(8)) + [9, 8]).tensor
        return tn.Node(A.tensor.reshape([4] * 5))
    A = tn.Node(tensor.reshape([d] * 4 + [d ** 2]))
    return A


def block_contribution(filename, h, n, corner_num, gap_l, edge_dim=2, replica=2, d=2, purity_mode=False):
    continuous_l = int(n / corner_num)
    w = corner_num * (continuous_l + gap_l)
    h = 2
    [cUp, dUp, cDown, dDown, leftRow, rightRow, openA, openB, A, B] = \
        get_boundaries_from_file(filename, w=2, h=2)
    normalization = pe.applyLocalOperators(cUp, dUp, cDown, dDown, leftRow, rightRow, A, B, h, w,
                                           [tn.Node(np.eye(d ** 2))] * h * w)
    leftRow.tensor /= normalization
    corner_charges_num = d ** corner_num
    wall_charges_num = d ** (2 * corner_num)
    boundaries_num = d ** (2 * n - corner_num)
    sum = 0
    for bi in range(boundaries_num):
        for cci in range(corner_charges_num):
            for wci in range(wall_charges_num):
                if bi == 0 and cci == 1 and wci == 3:
                    dbg = 1
                sum += np.abs(get_block_probability(filename, h, n, bi, d, corner_num, cci, wci, gap_l, PBC=False,
                                                    purity_mode=purity_mode,
                                                    boundaries=[cUp, dUp, cDown, dDown, leftRow, rightRow, openA,
                                                                openB])) ** replica
    return sum


def block_division_contribution(filename, h, n, corner_num, gap_l, edge_dim=2, replica=2, d=2):
    sum = block_contribution(filename, h, n, corner_num, gap_l, edge_dim, replica, d)
    return sum


def get_full_purity(filename, h, n, corner_num, gap_l, edge_dim=2, d=2, corner_charge=None, PBC=False, period_num=None):
    continuous_l = int(n / corner_num)
    w = corner_num * (continuous_l + gap_l)
    [cUp, dUp, cDown, dDown, leftRow, rightRow, openA, openB, A, B] = \
        get_boundaries_from_file(filename, w=2, h=2)
    normalization = pe.applyLocalOperators(cUp, dUp, cDown, dDown, leftRow, rightRow, A, B, h, w,
                                           [tn.Node(np.eye(d ** 2))] * h * w, PBC=PBC, period_num=period_num)
    leftRow.tensor /= (normalization**(2/h))
    if corner_charge is not None:
        projectors = [np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]),
                      np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]])]
        single_period_ops = [tn.Node(np.eye(d**2))] * (h - 1) + \
                            [tn.Node(projectors[corner_charge])] + \
                            [tn.Node(np.eye(d**2))] * (h * (continuous_l - 1)) + \
                            [tn.Node(np.eye(d**2))] * h * gap_l
        p1 = pe.applyLocalOperators(cUp, dUp, cDown, dDown, leftRow, rightRow, openA, openA, h, w, single_period_ops * corner_num)
    cUp, dUp, cDown, dDown, leftRow, rightRow, openA = [tn.Node(np.kron(o.tensor, o.tensor)) for o in
                                                        [cUp, dUp, cDown, dDown, leftRow, rightRow, openA]]
    if corner_charge is not None:
        corner_op = tn.Node(np.matmul(swap_op_tensor,
                    np.kron(projectors[corner_charge], projectors[corner_charge])))
    else:
        corner_op = swap_op
    single_period_ops = [swap_op] * (h - 1) + [corner_op] + [swap_op] * (h * (continuous_l - 1)) + \
                        [tn.Node(np.eye(4 ** 2))] * h * gap_l
    ops = []
    for ci in range(corner_num):
        ops = ops + single_period_ops
    purity = pe.applyLocalOperators(cUp, dUp, cDown, dDown, leftRow, rightRow, openA, openA, h, w, ops, PBC=PBC, period_num=period_num)
    if corner_charge is not None:
        purity /= p1**2
    if PBC:
        purity /= normalization**2
    return purity


def get_corner_projector(b, d=2):
    if d == 2:
        if b == 0:
            # 0000, 0101, 1010, 1111
            return tn.Node(np.diag([int(i in [0, 5, 10, 15]) for i in range(16)])
                           .reshape([4, 4, 4, 4]))
        else:
            # 0011, 0110, 1001, 1100
            return tn.Node(np.diag([int(i in [3, 6, 9, 12]) for i in range(16)])
                           .reshape([4, 4, 4, 4]))


corner_projector_0 = np.diag([1, 0, 0, 1])
corner_projector_1 = np.diag([0, 1, 1, 0])
corner_projectors = [corner_projector_0, corner_projector_1]
wall_projector_0 = np.diag([1, 0, 1, 0])
wall_projector_1 = np.diag([0, 1, 0, 1])
wall_projectors = [wall_projector_0, wall_projector_1]
swap_op_tensor = np.zeros((d, d, d, d))
for i in range(d):
    for j in range(d):
        swap_op_tensor[i, j, j, i] = 1
swap_op = tn.Node(swap_op_tensor.reshape([d**2, d**2]))


def get_block_probability(filename, h, n, bi, d=2, corner_num=1, corner_charges_i=0, wall_charges_i=0, gap_l=2, edge_dim=2,
                          PBC=False, purity_mode=False, normalize=False, boundaries=None):
    if boundaries is None:
        [cUp, dUp, cDown, dDown, leftRow, rightRow, openA, openB] = \
            get_boundaries(dir_name, model, param_name, param, silent=True)
    else:
        cUp, dUp, cDown, dDown, leftRow, rightRow, openA, openB = boundaries
    continuous_l = int(n / corner_num)
    if normalize:
        w = corner_num * (continuous_l + gap_l)
        norm = pe.applyLocalOperators(cUp, dUp, cDown, dDown, leftRow, rightRow,
                                      openA, openA, h, w, [tn.Node(np.eye(d ** 2))] * h * w, PBC=PBC)
        leftRow.tensor /= norm**(2/h)
    boundary = [int(c) for c in bin(bi).split('b')[1].zfill(2 * n - corner_num)]
    corner_charges = [int(c) for c in bin(corner_charges_i).split('b')[1].zfill(corner_num)]
    wall_charges = [int(c) for c in bin(wall_charges_i).split('b')[1].zfill(2 * corner_num * (h - 1))]
    edge_projectors = [tn.Node(np.diag([1, 0, 0, 0])), tn.Node(np.diag([0, 0, 0, 1]))]
    if edge_dim == 4:
        edge_projectors = [tn.Node(np.diag([1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1])),
                           tn.Node(np.diag([0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0]))]
    dUps = [bops.permute(bops.contract(dUp, edge_projectors[boundary[ni]], '1', '0'), [0, 2, 1]) for ni in
            range(int(n / 2))]
    cDowns = [bops.permute(bops.contract(cDown, edge_projectors[boundary[int(n / 2) + ni]], '1', '0'), [0, 2, 1]) for ni
              in range(int(n / 2))]
    cUps = [bops.permute(bops.contract(cUp, edge_projectors[boundary[int(n / 2) * 2 + ni]], '1', '0'), [0, 2, 1]) for ni
            in range(int(n / 2))]
    dDowns = [bops.permute(bops.contract(dDown, edge_projectors[boundary[int(n / 2) * 3 + ni]], '1', '0'), [0, 2, 1])
              for ni in range(int(n / 2) - corner_num)]
    ops = []
    cups_full = []
    dups_full = []
    cdowns_full = []
    ddowns_full = []
    for ci in range(corner_num):
        ops += [tn.Node(np.eye(d ** 2))] * h * (gap_l - 1) + \
               [tn.Node(wall_projectors[wall_charges[2 * (h - 1) * ci + hi]]) for hi in range(h - 1)] + \
               [tn.Node(np.eye(d ** 2))] + \
               [tn.Node(np.eye(d ** 2))] * (h - 1) + [tn.Node(corner_projectors[corner_charges[ci]])] + \
               [tn.Node(np.eye(d ** 2))] * h * (continuous_l - 2) + \
               [tn.Node(wall_projectors[wall_charges[2 * (h - 1) * ci + hi]]) for hi in range(h - 1, 2 * h - 2)] + \
               [tn.Node(np.eye(d ** 2))]
        cups_full += [cUp] * int(gap_l / 2) + cUps[int(continuous_l / 2) * ci: int(continuous_l / 2) * (ci + 1)]
        dups_full += [dUp] * int(gap_l / 2) + dUps[int(continuous_l / 2) * ci: int(continuous_l / 2) * (ci + 1)]
        cdowns_full += [cDown] * int(gap_l / 2) + cDowns[int(continuous_l / 2) * ci: int(continuous_l / 2) * (ci + 1)]
        ddowns_full += [dDown] * (int(gap_l / 2) + 1) + \
                       dDowns[(int(continuous_l / 2) - 1) * ci: (int(continuous_l / 2) - 1) * (ci + 1)]
    if purity_mode:
        dbg = 1
        cups_full = [tn.Node(np.kron(o.tensor, o.tensor)) for o in cups_full]
        dups_full = [tn.Node(np.kron(o.tensor, o.tensor)) for o in dups_full]
        cdowns_full = [tn.Node(np.kron(o.tensor, o.tensor)) for o in cdowns_full]
        ddowns_full = [tn.Node(np.kron(o.tensor, o.tensor)) for o in ddowns_full]
        leftRow = tn.Node(np.kron(leftRow.tensor, leftRow.tensor))
        rightRow = tn.Node(np.kron(rightRow.tensor, rightRow.tensor))
        openA = tn.Node(np.kron(openA.tensor, openA.tensor))
        ops = [tn.Node(np.kron(node.tensor, node.tensor)) for node in ops]
        for ci in range(corner_num):
            for si in range((gap_l + continuous_l) * h * ci + gap_l * h, (gap_l + continuous_l) * h * (ci + 1)):
                ops[si] = bops.contract(ops[si], swap_op, '1', '0')
    return pe.applyLocalOperators_detailedBoundary(cups_full, dups_full, cdowns_full, ddowns_full, [leftRow],
               [rightRow], openA, openA, 2, corner_num * (continuous_l + gap_l), ops, PBC=PBC, period_num=corner_num)


def get_purity(w, h, filename, boundary_ops=None, gap_l=2, PBC=False, corner_num=1):
    [cUp, dUp, cDown, dDown, leftRow, rightRow, openA, openB, A, B] = \
        get_boundaries_from_file(filename, w=w + gap_l * corner_num, h=h)
    cUp = tn.Node(np.kron(cUp.tensor, cUp.tensor))
    dUp = tn.Node(np.kron(dUp.tensor, dUp.tensor))
    cDown = tn.Node(np.kron(cDown.tensor, cDown.tensor))
    dDown = tn.Node(np.kron(dDown.tensor, dDown.tensor))
    leftRows = [tn.Node(np.kron(o.tensor, o.tensor)) for o in [leftRow] * int(h / 2)]
    rightRows = [tn.Node(np.kron(o.tensor, o.tensor)) for o in [rightRow] * int(h / 2)]
    openA = tn.Node(np.kron(openA.tensor, openA.tensor))
    openB = tn.Node(np.kron(openB.tensor, openB.tensor))
    ops = []
    cups_full = []
    dups_full = []
    cdowns_full = []
    ddowns_full = []
    continuous_l = int(w / corner_num)
    # TODO suit projectors to A and not openA, and for D=4
    double_corner_projectors = [np.kron(p, p) for p in corner_projectors]
    double_wall_projector_0 = np.kron(wall_projector_0, wall_projector_0)
    for ci in range(corner_num):
        ops += [tn.Node(np.eye(d ** 4))] * 2 * gap_l + \
               [tn.Node(swap_op)] * 2 * (continuous_l)
        cups_full += [cUp] * (int(gap_l / 2) + int(continuous_l / 2))
        dups_full += [dUp] * (int(gap_l / 2) + int(continuous_l / 2))
        cdowns_full += [cDown] * (int(gap_l / 2) + int(continuous_l / 2))
        ddowns_full += [dDown] * (int(gap_l / 2) + int(continuous_l / 2))
    purity = pe.applyLocalOperators_detailedBoundary(cups_full, dups_full, cdowns_full, ddowns_full,
                 leftRows, rightRows, openA, openB, h, (continuous_l + gap_l) * corner_num, ops, PBC=PBC,
                                                     period_num=corner_num)
    return purity


def numberToBase(n, b):
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return digits[::-1]


def tensors_from_transfer_matrix(model, param, d=2):
    A = toric_tensors_lgt_approach(model, param, d)
    E0 = bops.permute(bops.contract(A, A, '4', '4'), [0, 4, 1, 5, 2, 6, 3, 7])
    bond_dim = A[0].dimension
    projector_tensor = np.zeros([bond_dim] * 3)
    for i in range(d):
        projector_tensor[i, i, i] = 1
    projector = tn.Node(projector_tensor)
    tau = bops.contract(bops.contract(bops.contract(bops.contract(
        E0, projector, '01', '01'), projector, '01', '01'), projector, '01', '01'), projector, '01', '01')
    openA = tn.Node(np.kron(A.tensor, A.tensor.conj()) \
                    .reshape([bond_dim ** 2] * 4 + [d ** 2] * 2).transpose([4, 0, 1, 2, 3, 5]))
    return A, tau, openA, projector


def results_filename(dirname, model, param_name, param, Ns):
    return dirname + '/normalized_p2_results_' + model + '_' + param_name + '_' + str(param) + \
        '_Ns_' + str(Ns[0]) + '-' + str(Ns[-1])


def boundary_filname(dirname, model, param_name, param):
    return dirname + '/toricBoundaries_gauge_' + model + '_' + param_name + '_' + str(param)


def shrink_boundaries(upRow, downRow, leftRow, rightRow, bond_dim):
    max_te = 0
    [leftRow, upRow, te] = bops.svdTruncation(bops.contract(leftRow, upRow, '3', '0'), [0, 1, 2], [3, 4, 5], '>>',
                                              maxBondDim=bond_dim, minBondDim=bond_dim)
    if len(te) > 0 and max(te) > max_te: max_te = max(te)
    [downRow, leftRow, te] = bops.svdTruncation(bops.contract(downRow, leftRow, '3', '0'), [0, 1, 2], [3, 4, 5], '>>',
                                                maxBondDim=bond_dim, minBondDim=bond_dim)
    if len(te) > 0 and max(te) > max_te: max_te = max(te)
    [rightRow, downRow, te] = bops.svdTruncation(bops.contract(rightRow, downRow, '3', '0'), [0, 1, 2], [3, 4, 5], '>>',
                                                 maxBondDim=bond_dim, minBondDim=bond_dim)
    if len(te) > 0 and max(te) > max_te: max_te = max(te)
    [upRow, rightRow, te] = bops.svdTruncation(bops.contract(upRow, rightRow, '3', '0'), [0, 1, 2], [3, 4, 5], '>>',
                                               maxBondDim=bond_dim, minBondDim=bond_dim)
    if len(te) > 0 and max(te) > max_te: max_te = max(te)
    return upRow, downRow, leftRow, rightRow, max_te


def get_boundaries_corner_tm(dirname, model, param_name, param, h, w, chi):
    boundary_filename = boundary_filname(dirname, model, param_name, param) + '_corner'
    if os.path.exists(boundary_filename):
        [c_up_left, c_up_right, c_down_left, c_down_right, t_left, t_up, t_right, t_down, A, AEnv, openA] = \
            pickle.load(open(boundary_filename, 'rb'))
    else:
        A, tau, openA, singlet_projector = tensors_from_transfer_matrix(model, param, d=d)
        AEnv = bops.contract(openA, tn.Node(np.eye(4)), '05', '01')
        c_up_left_tensor = AEnv.tensor[0, :, :, 0].T
        c_up_right_tensor = AEnv.tensor[0, 0, :, :].T
        c_down_left_tensor = AEnv.tensor[:, :, 0, 0].T
        c_down_right_tensor = AEnv.tensor[:, 0, 0, :]
        left_edge_tensor = AEnv.tensor[:, :, :, 0].transpose([2, 1, 0])
        up_edge_tensor = AEnv.tensor[0, :, :, :].transpose([2, 1, 0])
        right_edge_tensor = AEnv.tensor[:, 0, :, :].transpose([0, 2, 1])
        down_edge_tensor = AEnv.tensor[:, :, 0, :].transpose([2, 0, 1])
        c_up_left, c_up_right, c_down_left, c_down_right, t_left, t_up, t_right, t_down = \
            ctm.corner_transfer_matrix(tn.Node(c_up_left_tensor), tn.Node(c_up_right_tensor),
                        tn.Node(c_down_left_tensor), tn.Node(c_down_right_tensor),
                        tn.Node(left_edge_tensor), tn.Node(up_edge_tensor),
                        tn.Node(right_edge_tensor), tn.Node(down_edge_tensor), AEnv, chi=chi)
    return c_up_left, c_up_right, c_down_left, c_down_right, t_left, t_up, t_right, t_down, A, AEnv, openA


def get_boundaries(dirname, model, param_name, param, max_allowed_te=1e-10, silent=False):
    boundary_filename = boundary_filname(dirname, model, param_name, param)
    if os.path.exists(boundary_filename):
        [upRow, downRow, leftRow, rightRow, openA, openA, A, A] = pickle.load(open(boundary_filename, 'rb'))
    else:
        A, tau, openA, singlet_projector = tensors_from_transfer_matrix(model, param, d=d)
        bond_dim = A[0].dimension
        AEnv = bops.contract(openA, tn.Node(np.eye(4)), '05', '01')
        upRow, downRow, leftRow, rightRow = peps.applyBMPS(AEnv, AEnv, d=d ** 2, gauge=True)
        with open(boundary_filename, 'wb') as f:
            pickle.dump([upRow, downRow, leftRow, rightRow, openA, openA, A, A], f)
    bond_dim = 2
    while True:
        upRow, downRow, leftRow, rightRow, max_te = shrink_boundaries(upRow, downRow, leftRow, rightRow, bond_dim)
        if max_te > max_allowed_te:
            bond_dim += 1
        else:
            break
    if not silent: print('truncation error: ' + str(max_te) + ', bond dim: ' + str(bond_dim))
    [cUp, dUp, te] = bops.svdTruncation(upRow, [0, 1], [2, 3], '>>', maxBondDim=upRow.tensor.shape[0])
    [cDown, dDown, te] = bops.svdTruncation(downRow, [0, 1], [2, 3], '>>', maxBondDim=downRow.tensor.shape[0])
    cUp.tensor = 0.5 * (cUp.tensor + cUp.tensor.transpose([2, 1, 0]))
    dUp.tensor = 0.5 * (dUp.tensor + dUp.tensor.transpose([2, 1, 0]))
    cDown.tensor = 0.5 * (cDown.tensor + cDown.tensor.transpose([2, 1, 0]))
    dDown.tensor = 0.5 * (dDown.tensor + dDown.tensor.transpose([2, 1, 0]))
    return cUp, dUp, cDown, dDown, leftRow, rightRow, openA, openA


# For debugging purposes - can erase when done
def split_kron(node):
    tensor = node.tensor
    sh = tensor.shape
    l, s, r, te = bops.svdTruncation(tn.Node(tensor.reshape([int(np.sqrt(sh[i//2])) for i in range(len(sh)*2)])
            .transpose([i * 2 for i in range(len(sh))] + [i * 2 + 1 for i in range(len(sh))])),
                       list(range(len(sh))), list(range(len(sh), 2 * len(sh))), '>*<')
    return l, s, r, te



def to_cannonical(psi, PBC=True):
    if PBC:
        N = len(psi)
        working_psi = [tn.Node(bops.contract(psi[0], psi[-1], '0', '2').tensor.transpose([0, 3, 1, 2])
                       .reshape([1, psi[0][1].dimension**2, psi[0][2].dimension * psi[-1][0].dimension]))] + \
                      [tn.Node(np.kron(psi[i].tensor, psi[-1-i].tensor.transpose([2, 1, 0]))) for i in range(1, N//2 - 1)] + \
                      [tn.Node(bops.contract(psi[N//2 - 1], psi[N//2], '2', '0').tensor.transpose([0, 3, 1, 2])
                       .reshape([psi[N//2 - 1][0].dimension * psi[N//2][2].dimension, psi[0][1].dimension**2, 1]))]
    else:
        working_psi = psi
    # for i in range(len(working_psi) - 1):
    #     working_psi[i], working_psi[i+1], te = \
    #         bops.svdTruncation(bops.contract(working_psi[i], working_psi[i+1], '2', '0'), [0, 1], [2, 3], '>>')
    return working_psi


def fold_mpo(mpo):
    N = len(mpo)
    return [tn.Node(bops.contract(mpo[0], mpo[-1], '2', '3').tensor.transpose([0, 3, 1, 4, 2, 5])
                    .reshape([mpo[0][0].dimension**2] * 2 + [1, mpo[0][3].dimension * mpo[-1][2].dimension]))] + \
           [tn.Node(np.kron(mpo[i].tensor, mpo[-1-i].tensor.transpose([0, 1, 3, 2]))) for i in range(1, N//2 - 1)] + \
           [tn.Node(bops.contract(mpo[N//2 - 1], mpo[N//2], '3', '2').tensor.transpose([0, 3, 1, 4, 2, 5])
                    .reshape([mpo[0][0].dimension**2] * 2 + [mpo[N//2 - 1][2].dimension * mpo[N//2][3].dimension, 1]))]


def large_system_expectation_value(w, h, cUp, dUp, cDown, dDown, leftRow, rightRow, openA, tau_projector, ops, chi=128):
    open_tau = bops.contract(bops.contract(bops.contract(bops.contract(
        openA, tau_projector, '1', '1'), tau_projector, '1', '1'), tau_projector, '1', '1'), tau_projector, '1', '1')
    p_c_up = bops.permute(bops.contract(cUp, tau_projector, '1', '1'), [0, 2, 1])
    p_d_up = bops.permute(bops.contract(dUp, tau_projector, '1', '1'), [0, 2, 1])
    p_c_down = bops.permute(bops.contract(cDown, tau_projector, '1', '1'), [1, 2, 0])
    p_d_down = bops.permute(bops.contract(dDown, tau_projector, '1', '1'), [1, 2, 0])
    full_system_length = 2 * w
    up_row = [p_c_up, p_d_up] * int(full_system_length / 2)
    up_row = to_cannonical(up_row, PBC=True)
    down_row = [p_d_down, p_c_down] * int(full_system_length / 2)
    down_row = to_cannonical(down_row, PBC=True)
    for hi in range(h):
        dbg = 1
        mid_row = [tn.Node(bops.permute(bops.contract(open_tau, ops[hi][wi], '01', '01'), [0, 2, 3, 1]))
                   for wi in range(w)] + \
            [tn.Node(bops.permute(bops.contract(open_tau, tn.Node(np.eye(openA[0].dimension)), '01', '01'),
                                  [0, 2, 3, 1]))] * (full_system_length - w) # w * cylinder_mult
        mid_row = fold_mpo(mid_row)
        if openA[0].dimension == 16:
            dbg = 1
        for wi in range(len(up_row)):
            up_row[wi] = tn.Node(bops.contract(up_row[wi], mid_row[wi], '1', '0').tensor.\
                transpose([0, 3, 2, 1, 4]).reshape(
                [up_row[wi][0].dimension * mid_row[wi][2].dimension,
                 mid_row[wi][1].dimension,
                 up_row[wi][2].dimension * mid_row[wi][3].dimension]))
        up_max_te = 0
        for k in range(len(up_row) - 2, -1, -1):
            M = bops.contract(up_row[k], up_row[(k+1) % len(up_row)], '2', '0')
            l, s, r, te = bops.svdTruncation(M, [0, 1], [2, 3], '>*<', maxBondDim=chi, normalize=False)
            up_row[k], up_row[(k+1) % len(up_row)] = bops.contract(l, s, '2', '0'), r
            tst = bops.contract(bops.contract(up_row[k], up_row[(k+1)], '2', '0'), M, '0123', '0123*').tensor \
                  / bops.contract(M, M, '0123', '0123*').tensor
            # print(hi, k, tst, openA[0].dimension)
            if np.abs(tst - 1) > 1e-3:
                dbg = 1
            if len(te) > 0 and np.max(te / np.max(s.tensor)) > 1e-3:
                print(np.diag(s.tensor), np.max(te) / np.max(s.tensor))
            if sum(np.diag(s.tensor)) < 1e-8:
                dbg = 1
                bops.svdTruncation(M, [0, 1], [2, 3], '>*<', maxBondDim=chi, normalize=False)
        for k in range(len(up_row) - 1):
            up_row = bops.shiftWorkingSite(up_row, k, '>>')
        dbg = 1
    curr = bops.contract(up_row[0], down_row[0], '01', '01')
    for i in range(1, len(up_row)):
        curr = bops.contract(bops.contract(curr, up_row[i], '0', '0'), down_row[i], '01', '01')
    return curr.tensor[0, 0]


def large_system_block_entanglement(model, param, param_name, dirname, h, w, b_inds, corner_charges, tau_projector=None, chi=128, corner_num=1, sys_h=None, sys_w=None):
    [cUp, dUp, cDown, dDown, leftRow, rightRow, openA, openB] = \
        get_boundaries(dir_name, model, param_name, param, silent=True)
    D = int(np.sqrt(openA[1].dimension))
    if tau_projector is None:
        tau_projector = tn.Node(np.zeros((D, D**2)))
        for Di in range(D):
            tau_projector.tensor[Di, Di * (D + 1)] = 1
    if sys_w is None:
        sys_w = w
    if sys_h is None:
        sys_h = h

    horiz_projs = [tn.Node(np.diag([1, 1, 0, 0])), tn.Node(np.diag([0, 0, 1, 1])), tn.Node(np.eye(4))]
    vert_projs = [tn.Node(np.diag([1, 0, 1, 0])), tn.Node(np.diag([0, 1, 0, 1])),  tn.Node(np.eye(4))]
    corner_projs = [tn.Node(np.diag([1, 0, 0, 1])), tn.Node(np.diag([0, 1, 1, 0])),  tn.Node(np.eye(4))]
    I = tn.Node(np.eye(openA[0].dimension))

    subsystem_sites = [[[hi, wi] for wi in \
                       range(1 + int(hi > (h - 1 - corner_num)) * (hi - (h - 1 - corner_num)), w)] \
                       for hi in range(h - 1)]
    subsystem_sites = [item for sublist in subsystem_sites for item in sublist]

    if corner_num == h - 1:
        ops = [[I, corner_projs[corner_charges[0]] + [horiz_projs[b_inds[wi]] for wi in range(2, w - 1)] +
                [bops.contract(vert_projs[b_inds[w]], horiz_projs[b_inds[w - 1]], '0', '1')]] + [I] * (sys_w - w)]
    else:
        ops = [[vert_projs[b_inds[0]]] +
               [horiz_projs[b_inds[wi]] for wi in range(1, w - 1)] +
               [bops.contract(vert_projs[b_inds[w]], horiz_projs[b_inds[w - 1]], '0', '1')] + [I] * (sys_w - w)]
    for hi in range(1, h - 1 - corner_num):
        ops.append([vert_projs[b_inds[w + 1 + hi * 2]]] + [I] * (w - 2) + [vert_projs[b_inds[w + 2 + hi * 2]]]  + [I] * (sys_w - w))
    for hi in range(h - 1 - corner_num, h - 1):
        ops.append([I] * (hi - (h - 1 - corner_num) + 1) + [corner_projs[corner_charges[hi]]]
                   + [I] * (w + h - 4 - hi - corner_num) + [vert_projs[b_inds[w + h - 2 - corner_num + hi]]]  + [I] * (sys_w - w))
    # ops.append([I] * (1 + corner_num) + [horiz_projs[b_inds[wi]]
    #                                   for wi in range(w + 2 * h - 3 - 2 * corner_num, w + 2 * h - 4 - 3 * corner_num + w)])
    ops.append([I] * (1 + corner_num) + [horiz_projs[b_inds[wi]]
                                      for wi in range(w + 2 * h - 3 - 2 * corner_num, w + 2 * h - 5 - 3 * corner_num + w)] + [I]  + [I] * (sys_w - w))
    ops += [[I] * sys_w] * (sys_h - h)

    # ops = [[I for wi in range(sys_w)] for hi in range(sys_h)]

    norm = large_system_expectation_value(
        sys_w, sys_h, cUp, dUp, cDown, dDown, leftRow, rightRow, openA, tau_projector, ops, chi=chi)
    openA /= np.abs((norm) ** (1 / (2 * sys_w * sys_h)))
    # norm = large_system_expectation_value(
    #     sys_w, sys_h, cUp, dUp, cDown, dDown, leftRow, rightRow, openA, tau_projector, ops, chi=chi)
    # print(param, w, h, corner_num, 'norm')

    compare = np.ones((h - 2, w - 2))
    for ci in range(corner_num - 1):
        compare[h - 2 - ci, :(corner_num - ci)] = 0
    block = np.zeros((2**((w-2)*(h-2)), 2**((w-2)*(h-2))), dtype=complex)
    for ini in range(2**((w-2)*(h-2))):
        in_inds = np.array([int(xi) for xi in bin(ini).split('b')[1].zfill((w-2)*(h-2))]).reshape([h-2, w-2])
        if np.amin(compare - in_inds) == -1:
            continue
        for outi in range(2**((w-2)*(h-2))):
            out_inds = np.array([int(xi) for xi in bin(outi).split('b')[1].zfill((w - 2) * (h - 2))]).reshape([h-2, w-2])
            if np.amin(compare - out_inds) == -1:
                continue
            ops_copy = [[tn.Node(ops[hi][wi].tensor) for wi in range(sys_w)] for hi in range(sys_h)]
            for plaq_hi in range(len(in_inds)):
                for plaq_wi in range(len(in_inds[0])):
                    ops_copy[plaq_hi][plaq_wi+1].tensor = np.matmul(
                        np.kron(np.eye(2), np.linalg.matrix_power(X, in_inds[plaq_hi, plaq_wi])),
                        np.matmul(ops_copy[plaq_hi][plaq_wi+1].tensor,
                        np.kron(np.eye(2), np.linalg.matrix_power(X, out_inds[plaq_hi, plaq_wi]))))
                    ops_copy[plaq_hi + 1][plaq_wi + 1].tensor = np.matmul(
                        np.kron(np.linalg.matrix_power(X, in_inds[plaq_hi, plaq_wi]), np.linalg.matrix_power(X, in_inds[plaq_hi, plaq_wi])),
                        np.matmul(ops_copy[plaq_hi + 1][plaq_wi + 1].tensor,
                                  np.kron(np.linalg.matrix_power(X, in_inds[plaq_hi, plaq_wi]), np.linalg.matrix_power(X, out_inds[plaq_hi, plaq_wi]))))
                    ops_copy[plaq_hi+1][plaq_wi+2].tensor = np.matmul(
                        np.kron(np.linalg.matrix_power(X, in_inds[plaq_hi, plaq_wi]), np.eye(2)),
                        np.matmul(ops_copy[plaq_hi+1][plaq_wi+2].tensor,
                        np.kron(np.linalg.matrix_power(X, out_inds[plaq_hi, plaq_wi]), np.eye(2))))
            block[ini, outi] = large_system_expectation_value(
                sys_w, sys_h, cUp, dUp, cDown, dDown, leftRow, rightRow, openA, tau_projector, ops, chi=chi)
    block = block / block.trace()
    return np.linalg.matrix_power(block, 2).trace()

    # d = ops[0][0][0].dimension
    # swap_op_tensor = np.zeros((d, d, d, d))
    # for i in range(d):
    #     for j in range(d):
    #         swap_op_tensor[i, j, j, i] = 1
    # swap_op = tn.Node(swap_op_tensor.reshape([d**2, d**2]))
    # tau_projector_2 = tn.Node(np.kron(tau_projector.tensor, tau_projector.tensor))
    # ops_2 = [[tn.Node(np.kron(ops[hi][wi].tensor, ops[hi][wi].tensor)) for wi in range(sys_w)] for hi in range(sys_h)]
    # cUp_2, dUp_2, cDown_2, dDown_2, leftRow_2, rightRow_2, openA_2 = \
    #     [tn.Node(np.kron(node.tensor, node.tensor)) for node in [cUp, dUp, cDown, dDown, leftRow, rightRow, openA]]
    # # TODO the term below should equal norm**2, but it doesn't
    # # norm_2 = large_system_expectation_value(
    # #     sys_w, sys_h, cUp_2, dUp_2, cDown_2, dDown_2, leftRow_2, rightRow_2, openA_2, tau_projector_2, ops_2, chi=chi)
    # for wi in range(1, w):
    #     for hi in range(h - 1):
    #         if [hi, wi] in subsystem_sites:
    #             ops_2[hi][wi] = bops.contract(ops_2[hi][wi], swap_op, '1', '0')
    # print(param, w, h, corner_num, 'p2')
    # p2 = large_system_expectation_value(
    #     sys_w, sys_h, cUp_2, dUp_2, cDown_2, dDown_2, leftRow_2, rightRow_2, openA_2, tau_projector_2, ops_2, chi=chi)
    # print(param, p2 / np.abs(norm**2))
    # return p2 / np.abs(norm**2)


def wilson_expectations(model, param, param_name, dirname, plot=False, chi=128):
    [cUp, dUp, cDown, dDown, leftRow, rightRow, openA, openB] = \
        get_boundaries(dir_name, model, param_name, param, silent=True)

    Ls = 2 * np.array(range(3, 7))
    wilson_expectations = np.zeros(len(Ls), dtype=complex)
    for Li in range(len(Ls)):
        L = Ls[Li]
        wilson_exp = square_wilson_loop_expectation_value(cUp, dUp, cDown, dDown, leftRow, rightRow, openA, L, chi=chi)
        wilson_expectations[Li] = wilson_exp
        gc.collect()
    p, residuals, _, _, _ = np.polyfit(Ls, np.log(wilson_expectations), 2, full=True)
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(Ls, np.log(wilson_expectations))
        plt.title(param_name + ' = ' + str(param))
        plt.show()
    return p[0]


def purity_corner_law(model, param, param_name, dirname, plot=False, chi=128, tau_projector=None):
    L = 5
    corners = list(range(1, L - 1))
    p2s = np.zeros(len(corners), dtype=complex)
    for ci in range(len(corners)):
        c = corners[ci]
        p2_filename = dirname + '/corner_' + param_name + '_' + str(param) + '_p2_c_' + str(c) + '_chi_' + str(chi)
        if os.path.exists(p2_filename):
            p2 = pickle.load(open(p2_filename, 'rb'))
        else:
            p2 = large_system_block_entanglement(model, param, param_name, dirname,
                            L, L, [0] * L**2 * 100, corner_charges=[0] * L, tau_projector=tau_projector, chi=chi, corner_num=c)
            pickle.dump(p2, open(p2_filename, 'wb'))
        p2s[ci] = p2
        gc.collect()
    p, residuals, _, _, _ = np.polyfit(corners, np.log(p2s), 2, full=True)
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(corners, p2s)
        plt.title('corner ' + param_name + ' = ' + str(param) + ', fit params = ' + str(p))
        plt.show()
    return p[1], p2s, corners


def purity_area_law(model, param, param_name, dirname, plot=False, chi=128, tau_projector=None):
    Ls = np.array(range(3, 6))
    p2s = np.zeros(len(Ls), dtype=complex)
    norms_p2s = np.zeros(len(Ls), dtype=complex)
    for Li in range(len(Ls)):
        L = Ls[Li]
        p2_filename = dirname + '/' + param_name + '_' + str(param) + '_p2_square_L_' + str(L) + '_chi_' + str(chi)
        if os.path.exists(p2_filename):
            p2 = pickle.load(open(p2_filename, 'rb'))
        else:
            p2 = large_system_block_entanglement(model, param, param_name, dirname,
                            L, L, [0] * L**2 * 100, corner_charges=[0] * L, tau_projector=tau_projector, chi=chi, corner_num=1, sys_h=Ls[-1], sys_w=Ls[-1])
            pickle.dump(p2, open(p2_filename, 'wb'))
        p2s[Li] = p2
        gc.collect()
    p, residuals, _, _, _ = np.polyfit(Ls - 1, np.log(p2s), 2, full=True)
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(Ls - 1, p2s)
        plt.title('area ' + param_name + ' = ' + str(param) + ', fit params = ' + str([np.format_float_positional(x, precision=4) for x in p]))
        plt.show()
    return p[0], p2s, Ls


def purity_stairs_law():
    model = 'orus'
    params = [np.round(0.1 * i, 8) for i in range(3, 15)]
    for pi in range(len(params)):
        param = params[pi]


#TODO equipartition check, some system that changes area and corners

model = sys.argv[1]
edge_dim = 2
params = []
param_name = ''
if model == 'toric_c':
    params = [0.5, 1.0, 1.5, 3.0] #[np.round(0.1 * i, 8) for i in range(1, 16)] #[np.round(0.1 * i, 8) for i in range(1, 40)] + [np.round(1 + 0.01 * i, 8) for i in range(-9, 10)]
    params.sort()
    param_name = 'c'
elif model == 'orus':
    # https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.113.257202
    params = [np.round(0.1 * i, 8) for i in range(3, 13)]
    param_name = 'g'
elif model == 'vary_alpha':
    param_name = 'alpha'
    params = [np.round(0.2 * a, 8) for a in range(-10, 11)]
    beta = float(sys.argv[1])
    gamma = float(sys.argv[2])
    delta = float(sys.argv[3])
    model = model + '_' + str(beta) + '_' + str(gamma) + '_' + str(delta)
elif model == 'vary_beta':
    param_name = 'beta'
    params = [np.round(0.1 * a, 8) for a in range(11)]
    alpha = float(sys.argv[1])
    gamma = float(sys.argv[2])
    delta = float(sys.argv[3])
    model = model + '_' + str(alpha) + '_' + str(gamma) + '_' + str(delta)
elif model == 'vary_ad':
    param_name = 'alpha'
    params = [np.round(0.2 * a, 8) for a in range(-10, 8)]
    beta = float(sys.argv[1])
    gamma = float(sys.argv[2])
    model = model + '_' + str(beta) + '_' + str(gamma)
elif model == 'vary_gamma':
    param_name = 'gamma'
    # params = [np.round(0.001 * a, 3) for a in range(30)] \
    #          + [np.round(0.2 * a, 8) for a in range(5)] + \
    # params = [np.round(1 + 0.001 * a, 8) for a in range(-100, -80)] \
    #          + [np.round(1 + 0.001 * a, 8) for a in range(40, 50)]
    params = [np.round(a * 0.01, 8) for a in range(6)] + [np.round(1 + 0.002 * a, 8) for a in list(range(-30, 1)) + list(range(5, 30))]
    params.sort()
    alpha = float(sys.argv[2])
    beta = float(sys.argv[3])
    delta = float(sys.argv[4])
    model = model + '_' + str(alpha) + '_' + str(beta) + '_' + str(delta)
elif model == 'zohar_non_symm':
    param_name = 'gamma'
    params = [np.round(a * 0.01, 8) for a in range(6)] + [np.round(1 + 0.002 * a, 8) for a in list(range(-10, 1)) + list(range(5, 10))]
    params.sort()
    alpha = 1.0
    beta_ur = 0.1
    beta_rd = 0.2
    beta_dl = 0.3
    beta_lu = 0.4
    delta = 0.95
    model = model + '_' + str(alpha) + '_' + str(beta_lu) + '_' + str(beta_dl) + '_' + str(beta_rd) + '_' + str(beta_ur) + '_' + str(delta)
dir_name = "results/gauge/" + model
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

L_p2s = np.zeros(len(params))
c_p2s = np.zeros(len(params))
for pi in range(len(params)):
    param = params[pi]
    print(param)
    # tau_projector = np.zeros((2, 4))
    # tau_projector[0, 0] = 1
    # tau_projector[2, 3] = 1
    # tau_projector[1, 1] = np.sqrt(1/2)
    # tau_projector[1, 2] = np.sqrt(1/2)
    tau_projector = np.eye(4)
    L_p2s[pi] = purity_area_law(model, param, param_name, dir_name, plot=False, chi=8, tau_projector=tn.Node(tau_projector))[0]
    c_p2s[pi] = purity_corner_law(model, param, param_name, dir_name, plot=False, chi=8, tau_projector=tn.Node(tau_projector))[0]
print(L_p2s)
print(c_p2s)

wilson_results = np.zeros(len(params))
for pi in range(len(params)):
    param = params[pi]
    print(param)
    wilson_filename = 'results/gauge/' + model + '/wilson_' + model + '_' + param_name + '_' + str(param)
    if os.path.exists(wilson_filename):
        bulk_coeff = pickle.load(open(wilson_filename, 'rb'))
    else:
        bulk_coeff = wilson_expectations(model, param, param_name, dir_name, chi=256)
        pickle.dump(bulk_coeff, open(wilson_filename, 'wb'))
    wilson_results[pi] = bulk_coeff


full_purity_large_systems = np.zeros(len(params))
for pi in range(len(params)):
    param = params[pi]
    res_filename = 'results/gauge/' + model + '/full_p2_large_system_' + model + '_' + param_name + '_' + str(param)
    if os.path.exists(res_filename):
        full_p2 = pickle.load(open(res_filename, 'rb'))
    else:
        L = 6
        full_p2 = large_system_block_entanglement(model, params[pi], param_name, dir_name, L, L, [-1] * 100, [-1] * L, chi=128)
        pickle.dump(full_p2, open(res_filename, 'wb'))
    full_purity_large_systems[pi] = full_p2
    print(param, full_p2, 'full p2')


import matplotlib.pyplot as plt
plt.plot(params, np.abs(wilson_results) * 10)
plt.plot(params, L_p2s)
plt.plot(params, c_p2s)
plt.plot(params, -np.log(full_purity_large_systems), '--')
plt.legend(['Wilson  * 10', r'areas', r'corners', r'$p_2$ full'])
plt.show()
