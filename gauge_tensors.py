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


def square_wilson_loop_expectation_value(cUp: tn.Node, dUp: tn.Node, cDown: tn.Node, dDown: tn.Node,
                                         leftRow: tn.Node, rightRow: tn.Node, openA: tn.Node, L: int, chi=128, sys_L=None):
    tau_projector = tn.Node(np.eye(openA[1].dimension))
    X = np.array([[0, 1], [1, 0]])
    I = np.eye(2)
    if sys_L is None:
        sys_L = L

    open_tau = bops.contract(bops.contract(bops.contract(bops.contract(
        openA, tau_projector, '1', '1'), tau_projector, '1', '1'), tau_projector, '1', '1'), tau_projector, '1', '1')
    p_c_up = bops.permute(bops.contract(cUp, tau_projector, '1', '1'), [0, 2, 1])
    p_d_up = bops.permute(bops.contract(dUp, tau_projector, '1', '1'), [0, 2, 1])
    p_c_down = bops.permute(bops.contract(cDown, tau_projector, '1', '1'), [1, 2, 0])
    p_d_down = bops.permute(bops.contract(dDown, tau_projector, '1', '1'), [1, 2, 0])
    p_left = bops.permute(bops.contract(bops.contract(leftRow, tau_projector, '1', '1'), tau_projector, '1', '1'),
                          [0, 2, 3, 1])
    p_right = bops.permute(bops.contract(bops.contract(rightRow, tau_projector, '1', '1'), tau_projector, '1', '1'),
                           [0, 2, 3, 1])

    tst_norm = large_system_expectation_value(4, 4, p_c_up, p_d_up, p_c_down, p_d_down, p_left, p_right, open_tau,
                                          [[tn.Node(np.eye(openA[0].dimension))] * L] * L, chi=chi)
    openA.tensor *= tst_norm**(-1 / (2*4**2))
    wilson = large_system_expectation_value(sys_L, sys_L, p_c_up, p_d_up, p_c_down, p_d_down, p_left, p_right, open_tau,
            [[tn.Node(np.kron(I, X))] * (L - 1) + [tn.Node(np.kron(I, I))] + [tn.Node(np.kron(I, I))] * (sys_L - L)] + \
            [[tn.Node(np.kron(X, I))] + [tn.Node(np.kron(I, I))] * (L - 2) + [tn.Node(np.kron(X, I))]  + [tn.Node(np.kron(I, I))] * (sys_L - L)] * (L - 2) + \
            [[tn.Node(np.kron(X, X))] + [tn.Node(np.kron(I, X))] * (L - 2) + [tn.Node(np.kron(X, I))] + [tn.Node(np.kron(I, I))] * (sys_L - L)] + \
                                            [[tn.Node(np.kron(I, I))] * sys_L] * (sys_L - L), chi=chi)
    norm = large_system_expectation_value(sys_L, sys_L, p_c_up, p_d_up, p_c_down, p_d_down, p_left, p_right, open_tau,
                                          [[tn.Node(np.eye(openA[0].dimension))] * sys_L] * sys_L, chi=chi)
    if norm == 0:
        large_system_expectation_value(sys_L, sys_L, p_c_up, p_d_up, p_c_down, p_d_down, p_left, p_right, open_tau,
                                       [[tn.Node(np.eye(openA[0].dimension))] * sys_L] * sys_L, chi=chi)
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
    if model[:10] == 'vary_gamma':
        gamma = param
        tensor = get_zohar_tensor(alpha, beta * np.ones(4), gamma, delta)
    elif model == 'zohar_gamma':
        tensor[1, 0, 1, 0, 1, 0] = param
        tensor[0, 1, 0, 1, 0, 1] = param
    elif model == 'orus':
        for i in range(d):
            for j in range(d):
                tensor[i, j, :, :, :, :] *= (1 + param) ** (i + j)
        tensor /= ((1 + param) ** 2)**0.75
    elif model == 'orus_expanded':
        for i in range(d):
            for j in range(d):
                tensor[i, j, :, :, :, :] *= (1 + param) ** (i + j)
        tensor /= ((1 + param) ** 2)**0.75
        expander_term = 0.5
        expander_single = np.zeros((2, 4), dtype=complex)
        expander_single[0, 0] = 1
        expander_single[0, 3] = expander_term
        expander_single[1, 1] = 1
        expander_single[1, 2] = expander_term
        expander_double = np.zeros((2, 2, 4, 4), dtype=complex)
        expander_double[0, 0, 0, 0] = 1
        expander_double[0, 0, 0, 3] = expander_term**(-1)
        expander_double[0, 0, 3, 0] = expander_term**(-1)
        expander_double[0, 0, 3, 3] = expander_term**(-2)
        expander_double[1, 1, 0, 0] = 1
        expander_double[1, 1, 0, 3] = expander_term**(-1)
        expander_double[1, 1, 3, 0] = expander_term**(-1)
        expander_double[1, 1, 3, 3] = expander_term**(-2)
        expander_double[0, 0, 1, 1] = 1
        expander_double[0, 0, 1, 2] = expander_term**(-1)
        expander_double[0, 0, 2, 1] = expander_term**(-1)
        expander_double[0, 0, 2, 2] = expander_term**(-2)
        expander_double[1, 1, 1, 1] = 1
        expander_double[1, 1, 1, 2] = expander_term**(-1)
        expander_double[1, 1, 2, 1] = expander_term**(-1)
        expander_double[1, 1, 2, 2] = expander_term**(-2)
        expander_double[0, 1, 0, 1] = 1
        expander_double[0, 1, 0, 2] = expander_term**(-1)
        expander_double[0, 1, 3, 1] = expander_term**(-1)
        expander_double[0, 1, 3, 2] = expander_term**(-2)
        expander_double[0, 1, 1, 0] = 1
        expander_double[0, 1, 1, 3] = expander_term**(-1)
        expander_double[0, 1, 2, 0] = expander_term**(-1)
        expander_double[0, 1, 2, 3] = expander_term**(-2)
        expander_double[1, 0, 0, 1] = 1
        expander_double[1, 0, 0, 2] = expander_term**(-1)
        expander_double[1, 0, 3, 1] = expander_term**(-1)
        expander_double[1, 0, 3, 2] = expander_term**(-2)
        expander_double[1, 0, 1, 0] = 1
        expander_double[1, 0, 1, 3] = expander_term**(-1)
        expander_double[1, 0, 2, 0] = expander_term**(-1)
        expander_double[1, 0, 2, 3] = expander_term**(-2)
        tensor = np.tensordot(np.tensordot(np.tensordot(
            tensor, expander_single, ([0], [0])), expander_single, ([0], [0])), expander_double, ([0,1], [0,1]))\
            .transpose([2, 3, 4, 5, 0, 1]).reshape([4, 4, 4, 4, 4])
        return tn.Node(tensor)

    elif model == 'my':
        tensor_base = np.zeros((2, 2, 2, 2, 2, 2), dtype=complex)
        tensor_base[0, 0, 0, 0, 0, 0] = 1
        tensor_base[0, 0, 1, 1, 0, 0] = 1
        tensor_base[0, 1, 0, 1, 0, 1] = 1
        tensor_base[0, 1, 1, 0, 0, 1] = 1
        tensor_base[1, 0, 0, 1, 1, 0] = 1
        tensor_base[1, 0, 1, 0, 1, 0] = 1
        tensor_base[1, 1, 0, 0, 1, 1] = 1
        tensor_base[1, 1, 1, 1, 1, 1] = 1
        expander_single = np.zeros((2, 4), dtype=complex)
        expander_single[0, 0] = 1
        expander_single[0, 3] = param
        expander_single[1, 1] = 1
        expander_single[1, 2] = param
        expander_double = np.zeros((2, 2, 4, 4), dtype=complex)
        expander_double[0, 0, 0, 0] = 1
        expander_double[0, 0, 0, 3] = param**(-1)
        expander_double[0, 0, 3, 0] = param**(-1)
        expander_double[0, 0, 3, 3] = param**(-2)
        expander_double[1, 1, 0, 0] = 1
        expander_double[1, 1, 0, 3] = param**(-1)
        expander_double[1, 1, 3, 0] = param**(-1)
        expander_double[1, 1, 3, 3] = param**(-2)
        expander_double[0, 0, 1, 1] = 1
        expander_double[0, 0, 1, 2] = param**(-1)
        expander_double[0, 0, 2, 1] = param**(-1)
        expander_double[0, 0, 2, 2] = param**(-2)
        expander_double[1, 1, 1, 1] = 1
        expander_double[1, 1, 1, 2] = param**(-1)
        expander_double[1, 1, 2, 1] = param**(-1)
        expander_double[1, 1, 2, 2] = param**(-2)
        expander_double[0, 1, 0, 1] = 1
        expander_double[0, 1, 0, 2] = param**(-1)
        expander_double[0, 1, 3, 1] = param**(-1)
        expander_double[0, 1, 3, 2] = param**(-2)
        expander_double[0, 1, 1, 0] = 1
        expander_double[0, 1, 1, 3] = param**(-1)
        expander_double[0, 1, 2, 0] = param**(-1)
        expander_double[0, 1, 2, 3] = param**(-2)
        expander_double[1, 0, 0, 1] = 1
        expander_double[1, 0, 0, 2] = param**(-1)
        expander_double[1, 0, 3, 1] = param**(-1)
        expander_double[1, 0, 3, 2] = param**(-2)
        expander_double[1, 0, 1, 0] = 1
        expander_double[1, 0, 1, 3] = param**(-1)
        expander_double[1, 0, 2, 0] = param**(-1)
        expander_double[1, 0, 2, 3] = param**(-2)
        tensor = np.tensordot(np.tensordot(np.tensordot(
            tensor_base, expander_single, ([0], [0])), expander_single, ([0], [0])), expander_double, ([0,1], [0,1]))\
            .transpose([2, 3, 4, 5, 0, 1]).reshape([4, 4, 4, 4, 4])
        return tn.Node(tensor)
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
        if param > 1: tensor /= param
    elif model[:7] == 'gamma_c':
        tensor = np.zeros([4] * 4 + [2] * 2, dtype=complex)
        global c
        gamma = param
        for top_left in range(2):
            for top_right in range(2):
                for bottom_left in range(2):
                    for bottom_right in range(2):
                        tensor[top_left*2 + top_right, top_right*2 + bottom_right,
                        bottom_left*2 + bottom_right, top_left*2 + bottom_left,
                        int(top_left != top_right), int(top_right != bottom_right)] = c**top_right \
                                        * gamma**(int(top_right+top_left+bottom_left+bottom_right == 2) +
                                              2 * int(top_right+top_left+bottom_left+bottom_right == 4))
    A = tn.Node(tensor.reshape([tensor.shape[0]] * 4 + [d ** 2]))
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


swap_op_tensor = np.zeros((d, d, d, d))
for i in range(d):
    for j in range(d):
        swap_op_tensor[i, j, j, i] = 1
swap_op = tn.Node(swap_op_tensor.reshape([d**2, d**2]))


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
        global d
        A, tau, openA, singlet_projector = tensors_from_transfer_matrix(model, param, d=d)




        return [0, 0, 0, 0, 0, 0, openA, openA]




        bond_dim = A[0].dimension
        AEnv = bops.contract(openA, tn.Node(np.eye(4)), '05', '01')
        if not tau_projector is None:
            AEnv = bops.contract(bops.contract(bops.contract(bops.contract(
                AEnv, tau_projector, '0', '1'), tau_projector, '0', '1'),
                tau_projector, '0', '1'), tau_projector, '0', '1')
        upRow, downRow, leftRow, rightRow = peps.applyBMPS(AEnv, AEnv, d=d ** 2, gauge=True)
        # TODO add non-projected left and down rows
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


def large_system_expectation_value(w, h, p_c_up, p_d_up, p_c_down, p_d_down, p_left, p_right, open_tau, ops, chi=128, pbc=False):
    if pbc:
        full_system_length = 2 * w
        up_row = [p_c_up, p_d_up] * int(full_system_length / 2)
        down_row = [p_d_down, p_c_down] * int(full_system_length / 2)
        up_row = to_cannonical(up_row, PBC=True)
        down_row = to_cannonical(down_row, PBC=True)
    else:
        full_system_length = w
        up_row = [p_c_up, p_d_up] * int(full_system_length / 2)
        down_row = [p_d_down, p_c_down] * int(full_system_length / 2)
        up_row = [tn.Node(np.eye(up_row[0][0].dimension).reshape([1] + [up_row[0][0].dimension] * 2))]\
                 + up_row + [tn.Node(np.eye(up_row[-1][2].dimension).reshape([up_row[0][0].dimension] * 2 + [1]))]
        down_row = [tn.Node(np.eye(down_row[0][0].dimension).reshape([1] + [down_row[0][0].dimension] * 2))]\
                + down_row + [tn.Node(np.eye(down_row[-1][2].dimension).reshape([down_row[0][0].dimension] * 2 + [1]))]
        ld, lu, te = bops.svdTruncation(p_left, [0, 1], [2, 3], '>>')
        lefts = [tn.Node(lu.tensor.reshape(list(lu.tensor.shape) + [1]).transpose([2, 0, 3, 1])),
                 tn.Node(ld.tensor.reshape(list(ld.tensor.shape) + [1]).transpose([2, 0, 3, 1]))]
        ru, rd, te = bops.svdTruncation(p_right, [0, 1], [2, 3], '>>')
        rights = [tn.Node(ru.tensor.reshape(list(ru.tensor.shape) + [1]).transpose([0, 2, 1, 3])),
                 tn.Node(rd.tensor.reshape(list(rd.tensor.shape) + [1]).transpose([0, 2, 1, 3]))]
    for hi in range(h):
        dbg = 1
        mid_row = [tn.Node(bops.permute(bops.contract(open_tau, tn.Node(np.eye(open_tau[0].dimension)), '01', '01'),
                                  [0, 2, 3, 1]))] * ((full_system_length - w) // 2) + \
                  [tn.Node(bops.permute(bops.contract(open_tau, ops[hi][wi], '01', '01'), [0, 2, 3, 1]))
                   for wi in range(w)] + \
                  [tn.Node(bops.permute(bops.contract(open_tau, tn.Node(np.eye(open_tau[0].dimension)), '01', '01'),
                                  [0, 2, 3, 1]))] * (full_system_length - w - (full_system_length - w) // 2)
        if pbc:
            mid_row = fold_mpo(mid_row)
        else:
            mid_row = [lefts[hi % 2]] + mid_row + [rights[hi % 2]]
        for wi in range(len(up_row)):
            try:
                up_row[wi] = tn.Node(bops.contract(up_row[wi], mid_row[wi], '1', '0').tensor.\
                    transpose([0, 3, 2, 1, 4]).reshape(
                    [up_row[wi][0].dimension * mid_row[wi][2].dimension,
                     mid_row[wi][1].dimension,
                     up_row[wi][2].dimension * mid_row[wi][3].dimension]))
            except ValueError:
                dbg = 1
        for k in range(len(up_row) - 1):
            up_row, te = bops.shiftWorkingSite(up_row, k, '>>', maxBondDim=chi, return_trunc_err=True)
            if len(te) > 0 and np.max(te) > 1e-5:
                bops.shiftWorkingSite(up_row, k, '>>', maxBondDim=chi, return_trunc_err=True)
                l, s, r, te_ = bops.svdTruncation(bops.contract(up_row[k], up_row[k+1], '2', '0'), [0, 1], [2, 3], '>*<', maxBondDim=chi)
                # print(bops.contract(bops.contract(up_row[k], up_row[k+1], '2', '0'),
                #                     bops.contract(up_row[k], up_row[k+1], '2', '0'), '0123', '0123').tensor)
                if np.max(te) / np.max(np.diag(s.tensor)) > 1e-3:
                    print(np.max(te), np.max(te) / np.max(np.diag(s.tensor)))
                    dbg = 1
    curr = bops.contract(up_row[0], down_row[0], '01', '01')
    for i in range(1, len(up_row)):
        curr = bops.contract(bops.contract(curr, up_row[i], '0', '0'), down_row[i], '01', '01')
    return curr.tensor[0, 0]


def get_ops(openA, w, h, b_inds, corner_num, corner_charges, sys_w, sys_h):
    horiz_projs = [tn.Node(np.diag([1, 1, 0, 0])), tn.Node(np.diag([0, 0, 1, 1])), tn.Node(np.eye(4))]
    vert_projs = [tn.Node(np.diag([1, 0, 1, 0])), tn.Node(np.diag([0, 1, 0, 1])),  tn.Node(np.eye(4))]
    corner_projs = [tn.Node(np.diag([1, 0, 0, 1])), tn.Node(np.diag([0, 1, 1, 0])),  tn.Node(np.eye(4))]
    I = tn.Node(np.eye(openA[0].dimension))

    if corner_num == h - 1:
        ops = [[I, corner_projs[corner_charges[0]] + [horiz_projs[b_inds[wi]] for wi in range(2, w - 1)] +
                [bops.contract(vert_projs[b_inds[w]], horiz_projs[b_inds[w - 1]], '0', '1')]] + [I] * (sys_w - w)]
    else:
        ops = [[vert_projs[b_inds[0]]] +
               [horiz_projs[b_inds[wi]] for wi in range(1, w - 1)] +
               [bops.contract(vert_projs[b_inds[w]], horiz_projs[b_inds[w - 1]], '0', '1')] + [I] * (sys_w - w)]
    for hi in range(1, h - 1 - corner_num):
        ops.append([vert_projs[b_inds[w - 1 + hi * 2]]] + [I] * (w - 2) + [vert_projs[b_inds[w + hi * 2]]]  + [I] * (sys_w - w))
    for hi in range(h - 1 - corner_num, h - 1):
        ops.append([I] * (hi - (h - 1 - corner_num) + 1) + [corner_projs[corner_charges[hi]]]
                   + [I] * (w + h - 4 - hi - corner_num) + [vert_projs[b_inds[w + h - 2 - corner_num + hi]]]  + [I] * (sys_w - w))
    # ops.append([I] * (1 + corner_num) + [horiz_projs[b_inds[wi]]
    #                                   for wi in range(w + 2 * h - 3 - 2 * corner_num, w + 2 * h - 4 - 3 * corner_num + w)])
    ops.append([I] * (1 + corner_num) + [horiz_projs[b_inds[wi]]
                                      for wi in range(w + 2 * h - 2 - 2 * corner_num, w + 2 * h - 4 - 3 * corner_num + w + 1)] + [I] * (sys_w - w))
    ops += [[I] * sys_w] * (sys_h - h)
    return ops


def d_equals_D_block_entanlgement(model, param, param_name, h, w, b_inds, corner_charges, corner_num=1):
    [cUp, dUp, cDown, dDown, leftRow, rightRow, openA, openB] = \
        get_boundaries(dir_name, model, param_name, param, silent=True)

    D = int(np.sqrt(openA[1].dimension))
    tau_projector = tn.Node(np.zeros((D, D**2)))
    for Di in range(D):
        tau_projector.tensor[Di, Di * (D + 1)] = 1

    open_tau = bops.contract(bops.contract(bops.contract(bops.contract(
        openA, tau_projector, '1', '1'), tau_projector, '1', '1'), tau_projector, '1', '1'), tau_projector, '1', '1')
    if param > 1:
        open_tau.tensor /= param
    p_c_up = bops.permute(bops.contract(cUp, tau_projector, '1', '1'), [0, 2, 1])
    p_d_up = bops.permute(bops.contract(dUp, tau_projector, '1', '1'), [0, 2, 1])
    ups = [p_c_up, p_d_up]
    p_c_down = bops.permute(bops.contract(cDown, tau_projector, '1', '1'), [1, 2, 0])
    p_d_down = bops.permute(bops.contract(dDown, tau_projector, '1', '1'), [1, 2, 0])
    downs = [p_c_down, p_d_down]
    p_left = bops.permute(bops.contract(bops.contract(leftRow, tau_projector, '1', '1'), tau_projector, '1', '1'),
                          [0, 2, 3, 1])
    p_right = bops.permute(bops.contract(bops.contract(rightRow, tau_projector, '1', '1'), tau_projector, '1', '1'),
                           [0, 2, 3, 1])
    AEnv = bops.contract(open_tau, tn.Node(np.eye(4)), '01', '01')

    corner_proj_0 = np.zeros((4, 4, 2, 2))
    corner_proj_1 = np.zeros((4, 4, 2, 2))
    corner_proj_none = np.eye(16).reshape([4, 4, 4, 4]) #.transpose([0, 2, 1, 3])
    for i in range(2):
        for j in range(2):
            corner_proj_0[2*i + j, 2*i + j, i, j] = 1
            corner_proj_1[2 * i + j, 2 * ((i+1)%2) + (j+1)%2, i, j] = 1
    corner_projs = [tn.Node(corner_proj_0), tn.Node(corner_proj_1), tn.Node(corner_proj_none)]
    A_corners = [bops.contract(bops.contract(bops.contract(bops.contract(
        openA, tn.Node(np.eye(4)), '05', '01'), tau_projector, '0', '1'), tau_projector, '0', '1'),
        corner_projs[corner_charges[ci]], '01', '01') for ci in range(corner_num)]

    outer = tn.Node(np.eye(p_c_up[0].dimension))
    for wi in range(w // 2):
        outer = bops.contract(outer, tn.Node(p_c_up.tensor[:, b_inds[2 * wi], :]), '1', '0')
        outer = bops.contract(outer, tn.Node(p_d_up.tensor[:, b_inds[2 * wi + 1], :]), '1', '0')
    for hi in range(h // 2):
        outer = bops.contract(outer, tn.Node(p_right.tensor[:, b_inds[w + 2 * hi], b_inds[w + 1 + 2 * hi], :]), '1', '0')
    bottom_length = w - corner_num
    for wi in range(bottom_length // 2):
        outer = bops.contract(outer, tn.Node(p_c_down.tensor[:, b_inds[w + h + 2 * wi], :]), '1', '0')
        outer = bops.contract(outer, tn.Node(p_d_down.tensor[:, b_inds[w + h + 2 * wi + 1], :]), '1', '0')
    if bottom_length % 2 == 1:
        outer = bops.contract(outer, tn.Node(p_c_down.tensor[:, b_inds[w + h + bottom_length - 1], :]), '1', '0')
    for hi in range(bottom_length // 2):
        outer = bops.contract(tn.Node(p_left.tensor[:, b_inds[w + h + bottom_length + 2 * hi],
                                                b_inds[w + h + bottom_length + 2 * hi + 1], :]), outer, '1', '0')
    if bottom_length % 2 == 1:
        outer = bops.contract(tn.Node(p_left.tensor[:, :, b_inds[w + h + bottom_length + bottom_length], :]), outer, '2', '0')
    columns = []
    if corner_num == 7:
        dbg = 1
    for wi in range(w - bottom_length):
        columns.append(tn.Node(downs[(w - bottom_length + 1) % 2].tensor.transpose([1, 0, 2])))
        for hi in range(wi):
            columns[wi] = bops.contract(columns[wi], AEnv, [hi*2], '2')
        columns[wi] = tn.Node(columns[wi].tensor.transpose(list(range(2 * wi)) + [2 * wi + 1, 2 * wi + 2, 2 * wi]))
    if corner_num > 2:
        columns[w - bottom_length - 1] = bops.contract(columns[w - bottom_length - 1], p_left, '135', '012')
    for hi in range((corner_num - 3) // 2):
        columns[w - bottom_length - 1] = bops.contract(columns[w - bottom_length - 1], p_left,
                                                       [2 * corner_num - 2*(hi+1), 2 * (hi+2), 2 * (hi + 3)], '012')
    if corner_num % 2 == 1:
        stairs = columns[-1]
    else:
        if corner_num == 2:
            stairs = bops.contract(columns[-1], p_left, [1, 3], [0, 1])
        else:
            stairs = bops.contract(columns[-1], p_left,
                               [len(columns[-1].tensor.shape) - 1, len(columns[-1].tensor.shape) - 3], [0, 1])
    for wi in range(len(columns) - 2, -1, -1):
        stairs = bops.contract(columns[wi], stairs, [2 * i + 1 for i in range(len(columns[wi].tensor.shape) // 2)], list(range(len(columns[wi].tensor.shape) // 2)))
    if corner_num == 1:
        outer = bops.contract(outer, stairs, [0, 2], [1, 0])
    elif corner_num % 2 == 1:
        outer = bops.contract(outer, stairs, [0, 2], [len(stairs.tensor.shape) - 1, 0])
    else:
        outer = bops.contract(outer, stairs, [0, 1], [len(stairs.tensor.shape) - 1, 0])
    outer_corner_proj0 = np.zeros((2, 2, 2, 2))
    outer_corner_proj0[0, 0, 0, 0] = 1
    outer_corner_proj0[1, 1, 1, 1] = 1
    outer_corner_proj1 = np.zeros((2, 2, 2, 2))
    outer_corner_proj1[0, 1, 0, 0] = 1
    outer_corner_proj1[1, 0, 1, 1] = 1
    outer_corner_projs = [tn.Node(outer_corner_proj0), tn.Node(outer_corner_proj1)]
    for ci in range(corner_num):
        outer = bops.contract(outer, outer_corner_projs[corner_charges[corner_num - 1 - ci]], '01', '01')

    left_length = h - corner_num
    rows = []
    rows.append(tn.Node(AEnv.tensor[b_inds[0], :, :, b_inds[w + h + bottom_length]]))
    for wi in range(1, w - 1):
        rows[0] = bops.contract(rows[0], tn.Node(AEnv.tensor[b_inds[wi], :, :, :]), [wi - 1], [2])
    rows[0] = bops.contract(rows[0], tn.Node(AEnv.tensor[b_inds[w-1], b_inds[w], :, :]), [w - 2], '1')
    for hi in range(1, left_length):
        rows[0] = bops.contract(rows[0], tn.Node(AEnv.tensor[:, :, :, b_inds[w + h + bottom_length + hi]]), '0', '0')
        for wi in range(1, w-1):
            rows[0] = bops.contract(rows[0], AEnv, [0, w-1], '03')
        rows[0] = bops.contract(rows[0], tn.Node(AEnv.tensor[:, b_inds[w + hi], :, :]), [0, w-1], '02')
    for hi in range(h - left_length):
        if w - hi == 0:
            rows.append(tn.Node(A_corners[hi].tensor[:, b_inds[w + left_length + hi], :, :]))
        else:
            rows.append(tn.Node(AEnv.tensor[:, b_inds[w + left_length + hi], :, :]))
            for wi in range(1, w - hi - 1):
                rows[hi + 1] = bops.contract(rows[hi + 1], AEnv, [2 * wi], '1')
            rows[hi + 1] = bops.contract(rows[hi + 1], A_corners[hi], [2 * wi], '1')
    inner = bops.contract(rows[1], rows[0], [2 * i for i in range(w)], list(range(w - 1, -1, -1)))
    for hi in range(2, len(rows)):
        inner = bops.contract(rows[hi], inner, [2 * i for i in range(w - hi + 1)], list(range(w - hi + 1)))
    for wi in range(bottom_length):
        inner.tensor = inner.tensor[b_inds[w + h + wi], :]
    inner = bops.permute(inner, [2*i for i in range(corner_num)] + [2*i + 1 for i in range(corner_num)])
    inner_rdm = inner.tensor.reshape([2 ** corner_num] * 2)
    outer = bops.permute(outer, [2*i for i in range(corner_num)] + [2*i + 1 for i in range(corner_num)])
    outer_rdm = outer.tensor.reshape([2 ** corner_num] * 2)
    return np.linalg.matrix_power(inner_rdm, 2).trace() / inner_rdm.trace() ** 2
    # return np.matmul(inner_rdm, np.matmul(outer_rdm, np.matmul(inner_rdm, outer_rdm))). trace() / \
    #        np.matmul(inner_rdm, outer_rdm).trace()**2


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

    AEnv = bops.contract(openA, tn.Node(np.eye(4)), '05', '01')
    # AEnv.tensor /= param + 0.5
    cUp = tn.Node(AEnv.tensor[0, :, :, :].transpose([2, 1, 0]) + AEnv.tensor[3, :, :, :].transpose([2, 1, 0]))
    dUp = tn.Node(AEnv.tensor[0, :, :, :].transpose([2, 1, 0]) + AEnv.tensor[3, :, :, :].transpose([2, 1, 0]))
    cDown = tn.Node(AEnv.tensor[:, :, 0, :].transpose([2, 0, 1]) + AEnv.tensor[:, :, 3, :].transpose([2, 0, 1]))
    dDown = tn.Node(AEnv.tensor[:, :, 0, :].transpose([2, 0, 1]) + AEnv.tensor[:, :, 3, :].transpose([2, 0, 1]))
    leftRow = tn.Node(bops.contract(AEnv, AEnv, '2', '0').tensor[:, :, 0, :, :, 0].transpose([3, 2, 1, 0]) +
                      bops.contract(AEnv, AEnv, '2', '0').tensor[:, :, 0, :, :, 3].transpose([3, 2, 1, 0]) +
                      bops.contract(AEnv, AEnv, '2', '0').tensor[:, :, 3, :, :, 0].transpose([3, 2, 1, 0]) +
                      bops.contract(AEnv, AEnv, '2', '0').tensor[:, :, 3, :, :, 3].transpose([3, 2, 1, 0]))
    rightRow = tn.Node(bops.contract(AEnv, AEnv, '2', '0').tensor[:, 0, :, 0, :, :].transpose([0, 1, 3, 2]) +
                       bops.contract(AEnv, AEnv, '2', '0').tensor[:, 0, :, 3, :, :].transpose([0, 1, 3, 2]) +
                       bops.contract(AEnv, AEnv, '2', '0').tensor[:, 3, :, 0, :, :].transpose([0, 1, 3, 2]) +
                       bops.contract(AEnv, AEnv, '2', '0').tensor[:, 3, :, 3, :, :].transpose([0, 1, 3, 2]))


    subsystem_sites = [[[hi, wi] for wi in \
                       range(1 + int(hi > (h - 1 - corner_num)) * (hi - (h - 1 - corner_num)), w)] \
                       for hi in range(h - 1)]
    subsystem_sites = [item for sublist in subsystem_sites for item in sublist]
    ops = get_ops(openA, w, h, b_inds, corner_num, corner_charges, sys_w, sys_h)

    open_tau = bops.contract(bops.contract(bops.contract(bops.contract(
        openA, tau_projector, '1', '1'), tau_projector, '1', '1'), tau_projector, '1', '1'), tau_projector, '1', '1')
    p_c_up = bops.permute(bops.contract(cUp, tau_projector, '1', '1'), [0, 2, 1])
    p_d_up = bops.permute(bops.contract(dUp, tau_projector, '1', '1'), [0, 2, 1])
    p_c_down = bops.permute(bops.contract(cDown, tau_projector, '1', '1'), [1, 2, 0])
    p_d_down = bops.permute(bops.contract(dDown, tau_projector, '1', '1'), [1, 2, 0])
    p_left = bops.permute(bops.contract(bops.contract(leftRow, tau_projector, '1', '1'), tau_projector, '1', '1'),
                          [0, 2, 3, 1])
    p_right = bops.permute(bops.contract(bops.contract(rightRow, tau_projector, '1', '1'), tau_projector, '1', '1'),
                           [0, 2, 3, 1])

    norm = large_system_expectation_value(
        sys_w, sys_h, p_c_up, p_d_up, p_c_down, p_d_down, p_left, p_right, open_tau, ops, chi=chi)
    print('--', norm)
    open_tau.tensor /= np.abs(norm ** (1 / (2 * sys_w * sys_h)))
    p_c_up.tensor /= np.abs(norm**(1/(4*sys_w)))
    p_d_up.tensor /= np.abs(norm**(1/(4*sys_w)))
    p_c_down.tensor /= np.abs(norm**(1/(4*sys_w)))
    p_d_down.tensor /= np.abs(norm**(1/(4*sys_w)))
    norm = large_system_expectation_value(
        sys_w, sys_h, p_c_up, p_d_up, p_c_down, p_d_down, p_left, p_right, open_tau, ops, chi=chi)

    print('--', norm)
    d = ops[0][0][0].dimension
    swap_op_tensor = np.zeros((d, d, d, d))
    for i in range(d):
        for j in range(d):
            swap_op_tensor[i, j, j, i] = 1
    swap_op = tn.Node(swap_op_tensor.reshape([d**2, d**2]))
    ops_2 = [[tn.Node(np.kron(ops[hi][wi].tensor, ops[hi][wi].tensor)) for wi in range(len(ops[0]))] for hi in range(len(ops))]
    p_c_up_2, p_d_up_2, p_c_down_2, p_d_down_2, p_left_2, p_right_2, open_tau_2 = \
        [tn.Node(np.kron(node.tensor, node.tensor)) for node in [p_c_up, p_d_up, p_c_down, p_d_down, p_left, p_right, open_tau]]
    # TODO the term below should equal norm**2, but it doesn't
    norm_2 = large_system_expectation_value(
        sys_w, sys_h, p_c_up_2, p_d_up_2, p_c_down_2, p_d_down_2, p_left_2, p_right_2, open_tau_2, ops_2, chi=chi)
    norm_2 /= norm**2
    print(f"{norm_2=}")
    for wi in range(1, w):
        for hi in range(h - 1):
            if [hi, wi] in subsystem_sites:
                ops_2[hi][wi] = bops.contract(ops_2[hi][wi], swap_op, '1', '0')
    p2 = large_system_expectation_value(
        sys_w, sys_h, p_c_up_2, p_d_up_2, p_c_down_2, p_d_down_2, p_left_2, p_right_2, open_tau_2, ops_2, chi=chi)
    print('p2', p2, p2 / np.abs(norm**2))
    return p2 / np.abs(norm**2)


def wilson_expectations(model, param, param_name, dirname, plot=False, chi=128):
    [cUp, dUp, cDown, dDown, leftRow, rightRow, openA, openB] = \
        get_boundaries(dir_name, model, param_name, param, silent=True)

    Ls = 2 * np.array(range(4, 9))
    wilson_expectations = np.zeros(len(Ls), dtype=complex)
    for Li in range(len(Ls)):
        L = Ls[Li]
        filename = dirname + '/wilson_' + param_name + '_' + str(param) + '_L_' + str(L)
        if os.path.exists(filename):
            wilson_exp = pickle.load(open(filename, 'rb'))
        else:
            wilson_exp = square_wilson_loop_expectation_value(cUp, dUp, cDown, dDown, leftRow, rightRow, openA, L, chi=chi, sys_L = Ls[-1])
            pickle.dump(wilson_exp, open(filename, 'wb'))
        wilson_expectations[Li] = wilson_exp
        gc.collect()
    p, residuals, _, _, _ = np.polyfit(Ls, np.log(wilson_expectations), 2, full=True)
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(Ls, np.log(wilson_expectations))
        plt.title(param_name + ' = ' + str(param) + ' ' + str(p))
        plt.show()
    return p[0]


def purity_corner_law(model, param, param_name, dirname, plot=False, chi=128, tau_projector=None, charges=None, case=''):
    L = 6
    corners = 2 * np.array(range(1, int(L/2))) # np.array(range(1, L))
    p2s = np.zeros(len(corners), dtype=complex)
    for ci in range(len(corners)):
        c = corners[ci]
        p2_filename = dirname + '/corner_' + param_name + '_' + str(param) + '_p2_c_' + str(c) + '_chi_' + str(chi) + '_' + case
        if os.path.exists(p2_filename):
            p2 = pickle.load(open(p2_filename, 'rb'))
        else:
            if charges is None:
                charges = [0] * L**2 * 1000
            p2 = large_system_block_entanglement(model, param, param_name, dirname,
                            L, L, charges, corner_charges=[0] * L, tau_projector=tau_projector, chi=chi, corner_num=c)
            # p2 = d_equals_D_block_entanlgement(model, param, param_name, L*4, L, charges, corner_charges=[0] * 1000, corner_num=c)
            pickle.dump(p2, open(p2_filename, 'wb'))
        p2s[ci] = p2
        gc.collect()
    p, residuals, _, _, _ = np.polyfit(corners, np.log(p2s), 2, full=True)
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(corners, -np.log(p2s))
        plt.title('corner ' + param_name + ' = ' + str(param) + ', fit params = ' + str(p))
        plt.show()
    return p[1], p2s, corners


def purity_area_law(model, param, param_name, dirname, plot=False, chi=128, tau_projector=None, b_inds=None, corner_charges=None):
    # Ls = np.array(range(3, 8))
    Ls = 2 * np.array(range(3, 7))
    p2s = np.zeros(len(Ls), dtype=complex)
    norms_p2s = np.zeros(len(Ls), dtype=complex)
    for Li in range(len(Ls)):
        L = Ls[Li]
        print(L)
        if b_inds is None:
            b_inds = [0] * L**2 * 100
            corner_charges = [0] * L * 100
        p2_filename = dirname + '/' + param_name + '_' + str(param) + '_p2_square_L_' + str(L) + '_chi_' + str(chi)
        if not b_inds is None:
            p2_filename += '_' + str(corner_charges[0])
        if os.path.exists(p2_filename):
            p2 = pickle.load(open(p2_filename, 'rb'))
        else:
            sys_size = 8
            if True: # b_inds[0] == -1:
                p2 = large_system_block_entanglement(model, param, param_name, dirname,
                                L, L, b_inds, corner_charges=corner_charges, tau_projector=tau_projector, chi=chi,
                                                     corner_num=1, sys_h=sys_size, sys_w=sys_size)
            else:
                p2 = d_equals_D_block_entanlgement(model, param, param_name, 2*L, L, b_inds, corner_charges, corner_num=1)
            pickle.dump(p2, open(p2_filename, 'wb'))
        p2s[Li] = p2
        gc.collect()
    p, residuals, _, _, _ = np.polyfit(Ls, np.log(p2s), 1, full=True)
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(Ls, -np.log(p2s))
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
    tau_projector = tn.Node(np.eye(4))
    params = [np.round(0.01 * i, 8) for i in range(1, 5)] + [np.round(0.1 * i, 8) for i in range(1, 20)] + [np.round(1 + 0.01 * i, 8) for i in range(-9, 10)]
    params.sort()
    param_name = 'c'
elif model == 'orus':
    # https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.113.257202
    params = [np.round(0.1 * i, 8) for i in range(3, 13)]
    param_name = 'g'
elif model == 'vary_gamma':
    param_name = 'gamma'
    tau_projector = None
    # params = [np.round(a * 0.01, 8) for a in range(6)] + [np.round(0.1 * a, 8) for a in range(1, 60)] +\
    #          [np.round(1.2 + 0.002 * a, 8) for a in list(range(-10, 10))]
    params = [np.round(0.1 * a, 8) for a in range(20)] + [np.round(1 + 0.01 * i, 8) for i in range(-10, 10)] + [np.round(0.01 * i, 8) for i in range(1, 10)]
    params.sort()
    alpha = float(sys.argv[2])
    beta = float(sys.argv[3])
    delta = float(sys.argv[4])
    model = model + '_' + str(alpha) + '_' + str(beta) + '_' + str(delta)
elif model == 'gamma_c':
    c = float(sys.argv[2])
    param_name = 'gamma'
    params = [float(sys.argv[3])] #[0.0, 0.4] + [np.round(0.2 * a, 8) for a in range(5, 10)]
    params.sort()
    tau_projector = None
    model = model + '_' + str(c)


dir_name = sys.argv[-1] + model # "results/gauge/" + model
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

L_p2s = np.zeros(len(params))
c_p2s = np.zeros(len(params))
c1_p2s = np.zeros(len(params))
c2_p2s = np.zeros(len(params))
for pi in range(len(params)):
    param = params[pi]
    print(param)
    L_p2s[pi] = purity_area_law(model, param, param_name, dir_name, plot=False, chi=128, tau_projector=tau_projector)[0]
    c_p2s[pi] = purity_corner_law(model, param, param_name, dir_name, plot=False, chi=128, tau_projector=tau_projector)[0]
    c1_p2s[pi] = purity_corner_law(model, param, param_name, dir_name, plot=False, chi=16, tau_projector=tau_projector,
                                   charges=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] + [0]*1000, case='flux')[0]
    c2_p2s[pi] = purity_corner_law(model, param, param_name, dir_name, plot=False, chi=16, tau_projector=tau_projector,
                                   charges=[1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1] + [0]*1000, case='flux2')[0]

full_purity_large_systems = np.zeros(len(params))
for pi in range(len(params)):
    param = params[pi]
    print(param)
    full_p2 = purity_area_law(model, param, param_name, dir_name, plot=False, chi=128, tau_projector=tau_projector, b_inds=[-1]*1000, corner_charges=[-1]*100)
    full_purity_large_systems[pi] = full_p2[0]

wilson_results = np.zeros(len(params))
for pi in range(len(params)):
    param = params[pi]
    print(param)
    bulk_coeff = wilson_expectations(model, param, param_name, dir_name, chi=256, plot=False)
    wilson_results[pi] = bulk_coeff

colors = ['#FFD700', '#FFB14E', '#FA8775', '#EA5F94', '#CD34B5', '#9D02D7', '#000099']

import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 1, height_ratios=[1, 2])
plt.subplots_adjust(hspace=0)
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

axs[0].plot(params, np.abs(wilson_results), color=colors[5])
axs[0].set_ylabel('confinement $c_w$', fontsize=14)
axs[1].set_xlabel(r'$\gamma$', fontsize=14)
axs[1].set_ylabel(r'$p_2$ system-size-dependence', fontsize=14)
axs[1].plot(params, L_p2s * 0.4, color=colors[2])
axs[1].plot(params, c_p2s, color=colors[6])
axs[1].plot(params, c1_p2s, '--', color=colors[1])
# axs[1].plot(params, c2_p2s, ':', color=colors[5])
axs[1].plot(params, full_purity_large_systems, color=colors[3])
axs[1].legend([r'SR Purity $c_{area}$', r'SR purity $c_{corner}, q=\{1,1\dots1\}$',
            r'SR purity $c_{corner}, q=$random',  r'full purity $c_{area}$'], fontsize=14)
plt.show()
