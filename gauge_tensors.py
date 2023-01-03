import numpy as np
import tensornetwork as tn
import basicOperations as bops
import pepsExpect as pe
import PEPS as peps
import pickle
import os
import sys
import gc


X = np.array([[0, 1], [1, 0]])
I = np.eye(2)
d=2


# |\psi> = \prod_p (1 + cX^p)|0>
# This construction does not obey Erez's requirement, but it has D=2
def get_toric_c_tensors(c):
    A_tensor = np.zeros((2, 2, 2, 2, 2, 2), dtype=complex)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                A_tensor[i, i, k, j, (i + j) % 2, (i + k) % 2] = c**i
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


def square_wilson_loop_expectation_value(cUp, dUp, cDown, dDown, leftRow, rightRow, openA, openB, L, d=2):
    w = int(np.ceil((L+1)/2)) * 2
    h = w
    I = np.eye(d)
    norm = pe.applyLocalOperators(cUp, dUp, cDown, dDown, leftRow, rightRow, openA, openA, h, w,
                                  [tn.Node(np.eye(d**2)) for i in range(w * h)])
    leftRow = bops.multNode(leftRow, 1 / norm**(2 / h))
    if L == 1:
        ops = [tn.Node(np.kron(I, X)), tn.Node(np.kron(X, X)),
               tn.Node(np.kron(I, I)), tn.Node(np.kron(X, I))]
    if L == 2:
        ops = [tn.Node(np.kron(I, I)), tn.Node(np.kron(I, X)), tn.Node(np.kron(X, I)), tn.Node(np.kron(X, X)),
               tn.Node(np.kron(I, I)), tn.Node(np.kron(I, X)), tn.Node(np.kron(I, I)), tn.Node(np.kron(I, X)),
               tn.Node(np.kron(I, I)), tn.Node(np.kron(I, I)), tn.Node(np.kron(X, I)), tn.Node(np.kron(X, I)),
               tn.Node(np.kron(I, I)), tn.Node(np.kron(I, I)), tn.Node(np.kron(I, I)), tn.Node(np.kron(I, I))]
    elif L == 3:
        ops = [tn.Node(np.kron(I, X)), tn.Node(np.kron(X, I)), tn.Node(np.kron(X, I)), tn.Node(np.kron(X, X)),
               tn.Node(np.kron(I, X)), tn.Node(np.kron(I, I)), tn.Node(np.kron(I, I)), tn.Node(np.kron(I, X)),
               tn.Node(np.kron(I, X)), tn.Node(np.kron(I, I)), tn.Node(np.kron(I, I)), tn.Node(np.kron(I, X)),
               tn.Node(np.kron(I, I)), tn.Node(np.kron(X, I)), tn.Node(np.kron(X, I)), tn.Node(np.kron(X, I))]
    elif L == 4:
        ops = [tn.Node(mat) for mat in [
            np.kron(I, I), np.kron(I, X), np.kron(X, I), np.kron(X, I), np.kron(X, I), np.kron(X, X),
            np.kron(I, I), np.kron(I, X), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, X),
            np.kron(I, I), np.kron(I, X), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, X),
            np.kron(I, I), np.kron(I, X), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, X),
            np.kron(I, I), np.kron(I, I), np.kron(X, I), np.kron(X, I), np.kron(X, I), np.kron(X, I),
            np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, I)
        ]]
    elif L == 5:
        ops = [tn.Node(mat) for mat in [
            np.kron(I, X), np.kron(X, I), np.kron(X, I), np.kron(X, I), np.kron(X, I), np.kron(X, X),
            np.kron(I, X), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, X),
            np.kron(I, X), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, X),
            np.kron(I, X), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, X),
            np.kron(I, X), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, X),
            np.kron(I, I), np.kron(X, I), np.kron(X, I), np.kron(X, I), np.kron(X, I), np.kron(X, I)
        ]]
    elif L == 6:
        ops = [tn.Node(mat) for mat in [
            np.kron(I, I), np.kron(I, X), np.kron(X, I), np.kron(X, I), np.kron(X, I), np.kron(X, I), np.kron(X, I), np.kron(X, X),
            np.kron(I, I), np.kron(I, X), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, X),
            np.kron(I, I), np.kron(I, X), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, X),
            np.kron(I, I), np.kron(I, X), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, X),
            np.kron(I, I), np.kron(I, X), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, X),
            np.kron(I, I), np.kron(I, X), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, X),
            np.kron(I, I), np.kron(I, I), np.kron(X, I), np.kron(X, I), np.kron(X, I), np.kron(X, I), np.kron(X, I), np.kron(X, I),
            np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, I)
        ]]
    elif L == 7:
        ops = [tn.Node(mat) for mat in [
            np.kron(I, X), np.kron(X, I), np.kron(X, I), np.kron(X, I), np.kron(X, I), np.kron(X, I), np.kron(X, I), np.kron(X, X),
            np.kron(I, X), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, X),
            np.kron(I, X), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, X),
            np.kron(I, X), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, X),
            np.kron(I, X), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, X),
            np.kron(I, X), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, X),
            np.kron(I, X), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, I), np.kron(I, X),
            np.kron(I, I), np.kron(X, I), np.kron(X, I), np.kron(X, I), np.kron(X, I), np.kron(X, I), np.kron(X, I), np.kron(X, I)
        ]]


    result = pe.applyLocalOperators(cUp, dUp, cDown, dDown, leftRow, rightRow, openA, openB, h, w, ops)
    return result, L**2, 4 * L


def get_boundaries_from_file(filename, w, h):
    with open(filename, 'rb') as f:
        [upRow, downRow, leftRow, rightRow, openA, openB, A, B] = pickle.load(f)
        [upRow, downRow, leftRow, rightRow, te] = shrink_boundaries(upRow, downRow, leftRow, rightRow, bond_dim=2)
        [cUp, dUp, te] = bops.svdTruncation(upRow, [0, 1], [2, 3], '>>', maxTrunc=5)
        [cDown, dDown, te] = bops.svdTruncation(downRow, [0, 1], [2, 3], '>>', maxTrunc=5)
        norm = pe.applyLocalOperators(cUp, dUp, cDown, dDown, leftRow, rightRow, A, B, h, w,
                                      [tn.Node(np.eye(4)) for i in range(w * h)])
        leftRow = bops.multNode(leftRow, 1 / norm**(2 / h))
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
    block = np.zeros((d**num_of_choices, d**num_of_choices), dtype=complex)
    choices = [[int(c) for c in bin(choice).split('b')[1].zfill(num_of_choices)] for choice in range(d**num_of_choices)]
    for ci in range(len(choices)):
        ingoing = get_choice_indices(boundary, choices[ci], n)
        for cj in range(len(choices)):
            outgoing = get_choice_indices(boundary, choices[cj], n)
            ops = [tn.Node(np.kron(projectors[ingoing[2 * i]][outgoing[2 * i]],
                           projectors[ingoing[2 * i + 1]][outgoing[2 * i + 1]])) for i in range(int(len(ingoing)/2))]
            block[ci, cj] = pe.applyLocalOperators(cUp, dUp, cDown, dDown, leftRow, rightRow, openA, openB, h=2, w=n, ops=ops)
    return block

def toric_tensors_lgt_approach(model, param, d=2):
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
    elif model == 'zohar_gamma':
        tensor[1, 0, 1, 0, 1, 0] = param
        tensor[0, 1, 0, 1, 0, 1] = param
    elif model == 'orus':
        for i in range(d):
            for j in range(d):
                tensor[i, j, :, :, :, :] *= (1 + param)**(i + j)
    elif model == 'zohar':
        alpha, beta, gamma, delta = param
        tensor[0, 0, 0, 0, 0, 0] = alpha
        for i in range(2):
            for j in range(2):
                tensor[i, j, (i + 1) % 2, (j + 1) % 2, i, j] = beta
        tensor[0, 1, 0, 1, 0, 1] = gamma
        tensor[1, 0, 1, 0, 1, 0] = gamma
        tensor[1, 1, 1, 1, 1, 1] = delta
    elif model == 'zohar_deltas':
        alpha, beta, gamma = np.sqrt(2), 1, 0
        delta = param
        tensor[0, 0, 0, 0, 0, 0] = alpha
        for i in range(2):
            for j in range(2):
                tensor[i, j, (i + 1) % 2, (j + 1) % 2, i, j] = beta
        tensor[0, 1, 0, 1, 0, 1] = gamma
        tensor[1, 0, 1, 0, 1, 0] = gamma
        tensor[1, 1, 1, 1, 1, 1] = delta
    elif model == 'zohar_deltas_large_alpha':
        alpha, beta, gamma = 2, 1, 0
        delta = param
        tensor[0, 0, 0, 0, 0, 0] = alpha
        for i in range(2):
            for j in range(2):
                tensor[i, j, (i + 1) % 2, (j + 1) % 2, i, j] = beta
        tensor[0, 1, 0, 1, 0, 1] = gamma
        tensor[1, 0, 1, 0, 1, 0] = gamma
        tensor[1, 1, 1, 1, 1, 1] = delta
    elif model == 'toric_c_mockup':
        tensor[0, 1, 0, 1, :, :] *= param**0.5
        tensor[1, 0, 1, 0, :, :] *= param**0.5
        tensor[1, 1, 0, 0, :, :] *= param**0.5
        tensor[0, 0, 1, 1, :, :] *= param**0.5
        tensor[1, 0, 0, 1, :, :] *= param**0.5
        tensor[0, 1, 1, 0, :, :] *= param**0.5
        tensor[1, 1, 1, 1, :, :] *= param
    elif model == 'toric_c':
        tensor = np.zeros([2] * 10, dtype=complex)
        tensor[0, 0, 0, 0, 0, 0, 0, 0, 0, 0] = 1
        A = tn.Node(tensor)
        x = tn.Node(X)
        A.tensor = A.tensor + param**0.25 * bops.contract(bops.contract(bops.permute(bops.contract(bops.contract(
            x, A, '1', '0'), x, '2', '1'), [0, 1, 9] + list(range(2, 9))), x, '8', '1'), x, '8', '1').tensor
        A.tensor = A.tensor + param**0.25 * bops.contract(bops.permute(bops.contract(bops.permute(bops.contract(
            A, x, '3', '1'), [0, 1, 2, 9] + list(range(3, 9))), x, '4', '1'), [0, 1, 2, 3, 9] + list(range(4, 9))),
            x, '9', '1').tensor
        A.tensor = A.tensor + param**0.25 * bops.permute(bops.contract(bops.permute(bops.contract(
            A, x, '5', '1'), list(range(5)) + [9, 5, 6, 7, 8]), x, '7', '1'), list(range(7)) + [9, 7, 8]).tensor
        A.tensor = A.tensor + param**0.25 * bops.permute(bops.contract(bops.permute(bops.contract(bops.permute(bops.contract(
            A, x, '1', '1'), [0, 9] + list(range(1, 9))), x, '7', '1'), list(range(7)) + [9, 7, 8]),
            x, '8', '1'), list(range(8)) + [9, 8]).tensor
        return tn.Node(A.tensor.reshape([4] * 5))
    A = tn.Node(tensor.reshape([d] * 4 + [d**2]))
    return A


def block_division_contribution(filename, n, corner_num, gap_l, edge_dim=2, replica=2, d=2):
    continuous_l = int(n / corner_num)
    w = corner_num * (continuous_l + gap_l)
    h = 2
    [cUp, dUp, cDown, dDown, leftRow, rightRow, openA, openB, A, B] = \
        get_boundaries_from_file(filename, w=2, h=2)
    normalization = pe.applyLocalOperators(cUp, dUp, cDown, dDown, leftRow, rightRow, A, B, h, w, [tn.Node(np.eye(d**2))] * h * w)
    corner_charges_num = d**corner_num
    wall_charges_num = d**(2 * corner_num)
    boundaries_num = d**(2 * n - corner_num)
    sum = 0
    for bi in range(boundaries_num):
        for cci in range(corner_charges_num):
            for wci in range(wall_charges_num):
                sum += get_block_probability(filename, n, bi, d, corner_num, cci, wci, gap_l, PBC=False) ** replica
    return sum / normalization**replica

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
swap_op_tensor = np.zeros((d**4, d**4))
for i in range(d**2):
    for j in range(d**2):
        swap_op_tensor[i + j * d**2, j + i * d**2] = 1
swap_op = tn.Node(swap_op_tensor)
def get_block_probability(filename, n, bi, d=2, corner_num=1, corner_charges_i=0, wall_charges_i=0, gap_l=2, edge_dim=2, PBC=False, purity_mode=False):
    [cUp, dUp, cDown, dDown, leftRow, rightRow, openA, openB, A, B] = \
        get_boundaries_from_file(filename, w=2, h=2)
    boundary = [int(c) for c in bin(bi).split('b')[1].zfill(2 * n - corner_num)]
    corner_charges = [int(c) for c in bin(corner_charges_i).split('b')[1].zfill(corner_num)]
    wall_charges = [int(c) for c in bin(wall_charges_i).split('b')[1].zfill(2 * corner_num)]
    edge_projectors = [tn.Node(np.diag([1, 0, 0, 0])), tn.Node(np.diag([0, 0, 0, 1]))]
    if edge_dim == 4:
        edge_projectors = [tn.Node(np.diag([1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1])),
                           tn.Node(np.diag([0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0]))]
    dUps = [bops.permute(bops.contract(dUp, edge_projectors[boundary[ni]], '1', '0'), [0, 2, 1]) for ni in range(int(n / 2))]
    cDowns = [bops.permute(bops.contract(cDown, edge_projectors[boundary[int(n/2) + ni]], '1', '0'), [0, 2, 1]) for ni in range(int(n / 2))]
    cUps = [bops.permute(bops.contract(cUp, edge_projectors[boundary[int(n/2) * 2 + ni]], '1', '0'), [0, 2, 1]) for ni in range(int(n / 2))]
    dDowns = [bops.permute(bops.contract(dDown, edge_projectors[boundary[int(n/2) * 3 + ni]], '1', '0'), [0, 2, 1]) for ni in range(int(n / 2) - corner_num)]
    ops = []
    cups_full = []
    dups_full = []
    cdowns_full = []
    ddowns_full = []
    continuous_l = int(n / corner_num)
    for ci in range(corner_num):
        ops += [tn.Node(np.eye(d**2))] * 2 * (gap_l - 1) + [tn.Node(wall_projectors[wall_charges[2 * ci]]), tn.Node(np.eye(d**2))] + \
               [tn.Node(np.eye(d**2)), tn.Node(corner_projectors[corner_charges[ci]])] + \
               [tn.Node(np.eye(d**2))] * 2 * (continuous_l - 2) + \
               [tn.Node(wall_projectors[wall_charges[2 * ci + 1]]), tn.Node(np.eye(d**2))]
        cups_full += [cUp] * int(gap_l / 2) + cUps[int(continuous_l/2) * ci: int(continuous_l/2) * (ci + 1)]
        dups_full += [dUp] * int(gap_l / 2) + dUps[int(continuous_l/2) * ci: int(continuous_l/2) * (ci + 1)]
        cdowns_full += [cDown] * int(gap_l / 2) + cDowns[int(continuous_l/2) * ci: int(continuous_l/2) * (ci + 1)]
        ddowns_full += [dDown] * (int(gap_l / 2) + 1) + dDowns[(int(continuous_l/2) - 1) * ci: (int(continuous_l/2) - 1) * (ci + 1)]
    if purity_mode:
        cups_full = [tn.Node(np.kron(node.tensor, node.tensor)) for node in cups_full]
        dups_full = [tn.Node(np.kron(node.tensor, node.tensor)) for node in dups_full]
        cdowns_full = [tn.Node(np.kron(node.tensor, node.tensor)) for node in cdowns_full]
        ddowns_full = [tn.Node(np.kron(node.tensor, node.tensor)) for node in ddowns_full]
        leftRow = tn.Node(np.kron(leftRow.tensor, leftRow.tensor))
        rightRow = tn.Node(np.kron(rightRow.tensor, rightRow.tensor))
        openA = tn.Node(np.kron(openA.tensor, openA.tensor))
        ops = [tn.Node(np.kron(node.tensor, node.tensor)) for node in ops]
        for ci in range(corner_num):
            for si in range((gap_l + continuous_l) * 2 * ci + gap_l * 2, (gap_l + continuous_l) * 2 * (ci + 1)):
                ops[si] = bops.contract(ops[si], swap_op, '1', '0')
    return pe.applyLocalOperators_detailedBoundary(cups_full, dups_full, cdowns_full, ddowns_full, [leftRow], [rightRow],
                openA, openA, 2, corner_num * (continuous_l + gap_l), ops, PBC=PBC, period_num=corner_num)


def get_purity(w, h, corner_num=1, corner_charges=[0], boundary_ops=None, gap_l=2):
    [cUp, dUp, cDown, dDown, leftRow, rightRow, openA, openB, A, B] = get_boundaries_from_file(filename, w=n + gap_l * (corner_num + 1), h=h)
    if boundary_ops is None:
        norm = pe.applyLocalOperators(cUp, dUp, cDown, dDown, leftRow, rightRow, openA, openB, w, h,
                                      [tn.Node(np.eye(4)) for i in range(w * h)])
        leftRow.tensor /= norm**(h / 2)
        cUps = [cUp for i in range(int(w/2))]
        dUps = [dUp for i in range(int(w/2))]
        cDowns = [cDown for i in range(int(w/2))]
        dDowns = [dDown for i in range(int(w/2))]
        leftRows = [leftRow for i in range(int(h/2))]
        rightRows = [rightRow for i in range(int(h/2))]
    else:
        cUps, dUps, cDowns, dDowns, leftRows, rightRows = boundary_ops
    cUp = tn.Node(np.kron(cUp.tensor, cUp.tensor))
    dUp = tn.Node(np.kron(dUp.tensor, dUp.tensor))
    cDown = tn.Node(np.kron(cDown.tensor, cDown.tensor))
    dDown = tn.Node(np.kron(dDown.tensor, dDown.tensor))
    cUps = [tn.Node(np.kron(o.tensor, o.tensor)) for o in cUps]
    dUps = [tn.Node(np.kron(o.tensor, o.tensor)) for o in dUps]
    cDowns = [tn.Node(np.kron(o.tensor, o.tensor)) for o in cDowns]
    dDowns = [tn.Node(np.kron(o.tensor, o.tensor)) for o in dDowns]
    leftRows = [tn.Node(np.kron(o.tensor, o.tensor)) for o in leftRows]
    rightRows = [tn.Node(np.kron(o.tensor, o.tensor)) for o in rightRows]
    A = tn.Node(np.kron(A.tensor, A.tensor))
    B = tn.Node(np.kron(B.tensor, B.tensor))
    single_swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    double_swap = np.kron(single_swap, single_swap).reshape([2] * 8).transpose([3, 2, 1, 0, 4, 5, 6, 7]).reshape(
        [2 ** 4, 2 ** 4])
    ops = []
    cups_full = []
    dups_full = []
    cdowns_full = []
    ddowns_full = []
    continuous_l = int(w / corner_num)
    # TODO suit projectors to A and not openA, and for D=4
    double_corner_projectors = [np.kron(p, p) for p in corner_projectors]
    double_wall_projector_0 = np.kron(wall_projector_0, wall_projector_0)
    for ci in range(1):
        ops += [tn.Node(np.eye(d**4))] * 2 * (gap_l - 1) + [tn.Node(double_wall_projector_0), tn.Node(np.eye(d**4))] + \
               [tn.Node(double_swap),
                tn.Node(np.matmul(double_swap, double_corner_projectors[corner_charges[ci]]))] + \
            [tn.Node(double_swap)] * 2 * (continuous_l - 2) + \
            [tn.Node(np.matmul(double_swap, double_wall_projector_0)), tn.Node(double_swap)]
        cups_full += [cUp] * int(gap_l / 2) + cUps[int(continuous_l/2) * ci: int(continuous_l/2) * (ci + 1)]
        dups_full += [dUp] * int(gap_l / 2) + dUps[int(continuous_l/2) * ci: int(continuous_l/2) * (ci + 1)]
        cdowns_full += [cDown] * int(gap_l / 2) + cDowns[int(continuous_l/2) * ci: int(continuous_l/2) * (ci + 1)]
        ddowns_full += [dDown] * int(gap_l / 2) + dDowns[int(continuous_l/2) * ci: int(continuous_l/2) * (ci + 1)]
    # ops += [tn.Node(np.eye(d**4))] * 2 * gap_l
    # cups_full += [cUp] * int(gap_l / 2)
    # dups_full += [dUp] * int(gap_l / 2)
    # cdowns_full += [cDown] * int(gap_l / 2)
    # ddowns_full += [dDown] * int(gap_l / 2)
    purity = pe.applyLocalOperators_detailedBoundary(cups_full, dups_full, cdowns_full, ddowns_full,
                                leftRows, rightRows, A, B, h, continuous_l + gap_l, ops, PBC=True, period_num=corner_num)
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
    openA = tn.Node(np.kron(A.tensor, A.tensor.conj())\
                    .reshape([bond_dim**2] * 4 + [d**2] * 2).transpose([4, 0, 1, 2, 3, 5]))
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
    return upRow, downRow,  leftRow, rightRow, max_te


# TODO handle A != B
def get_boundaries(dirname, model, param_name, param, max_allowed_te=1e-10, silent=False):
    boundary_filename = boundary_filname(dirname, model, param_name, param)
    if os.path.exists(boundary_filename):
        [upRow, downRow, leftRow, rightRow, openA, openA, A, A] = pickle.load(open(boundary_filename, 'rb'))
    else:
        A, tau, openA, singlet_projector = tensors_from_transfer_matrix(model, param, d=d)
        bond_dim = A[0].dimension
        singlet_projector = tn.Node(singlet_projector.tensor.reshape([bond_dim, bond_dim**2]))
        upRow, downRow, leftRow, rightRow = peps.applyBMPS(tau, tau, d=d**2)
        upRow = bops.permute(bops.contract(bops.contract(
            upRow, singlet_projector, '1', '0'), singlet_projector, '1', '0'), [0, 2, 3, 1])
        rightRow = bops.permute(bops.contract(bops.contract(
            rightRow, singlet_projector, '1', '0'), singlet_projector, '1', '0'), [0, 2, 3, 1])
        downRow = bops.permute(bops.contract(bops.contract(
            downRow, singlet_projector, '1', '0'), singlet_projector, '1', '0'), [0, 2, 3, 1])
        leftRow = bops.permute(bops.contract(bops.contract(
            leftRow, singlet_projector, '1', '0'), singlet_projector, '1', '0'), [0, 2, 3, 1])
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
    return cUp, dUp, cDown, dDown, leftRow, rightRow, openA, openA


def wilson_expectations(model, param, param_name, dirname, plot=False, d=2, boundaries=None):
    if boundaries is None:
        cUp, dUp, cDown, dDown, leftRow, rightRow, openA, openA = get_boundaries(dirname, model, param_name, param)
    else:
        cUp, dUp, cDown, dDown, leftRow, rightRow, openA, openA = boundaries

    Ls = np.array(range(2, 8))
    if openA[1].dimension > d**2:
        Ls = np.array(range(2, 6))
    perimeters = np.zeros(len(Ls))
    areas = np.zeros(len(Ls))
    wilson_expectations = np.zeros(len(Ls), dtype=complex)
    for Li in range(len(Ls)):
        print('L = ' + str(Ls[Li]))
        wilson_exp, area, perimeter = \
            square_wilson_loop_expectation_value(cUp, dUp, cDown, dDown, leftRow, rightRow, openA, openA, Ls[Li], d)
        perimeters[Li] = perimeter
        areas[Li] = area
        wilson_expectations[Li] = wilson_exp
        gc.collect()
    print(wilson_expectations)
    pa, residuals, _, _, _ = np.polyfit(areas, np.log(wilson_expectations), 1, full=True)
    chisq_dof = residuals / (len(areas) - 3)
    wilson_area = chisq_dof
    pp, residuals, _, _, _ = np.polyfit(perimeters, np.log(wilson_expectations), 1, full=True)
    chisq_dof = residuals / (len(perimeters) - 3)
    wilson_perimeter = chisq_dof
    if plot:
        import matplotlib.pyplot as plt
        ff, axs = plt.subplots(2)
        axs[0].scatter(areas, np.log(wilson_expectations))
        axs[0].plot(areas, areas * pa[0] + pa[1])
        axs[1].scatter(perimeters, np.log(wilson_expectations))
        axs[1].plot(perimeters, perimeters * pp[0] + pp[1])
        axs[0].set_title(param_name + ' = ' + str(param))
        plt.show()
    return wilson_area, wilson_perimeter


def normalized_p2s_data(model, params, Ns, dirname, param_name, d=2):
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    for pi in range(len(params)):
        param = params[pi]
        print(param)
        filename = results_filename(dirname, model, param_name, param, Ns)
        if os.path.exists(filename):
            continue
        boundary_filename = boundary_filname(dirname, model, param_name, param)

        wilson_area, wilson_perimeter = wilson_expectations(model, param, param_name, dirname)

        A, tau, openA, singlet_projector = tensors_from_transfer_matrix(model, param, d=d)
        tau = np.real(tau.tensor.transpose([0, 2, 1, 3]).reshape([d**2, d**2]))
        tau_eigenvals = np.zeros(d**2)
        tau_eigenvals[:2] = np.real(np.linalg.eigvals([[tau[0, 0], tau[0, 3]], [tau[3, 0], tau[3, 3]]]))
        tau_eigenvals[2:] = np.real(np.linalg.eigvals([[tau[1, 1], tau[1, 2]], [tau[2, 1], tau[2, 2]]]))
        num_of_sampled_blocks = 63
        sampled_blocks = [[int(i * d**(2 * n + 2) / num_of_sampled_blocks) for i in range(num_of_sampled_blocks)] for n in Ns]
        p2s = [np.zeros(num_of_sampled_blocks) for n in Ns]
        p1s = [np.zeros(num_of_sampled_blocks) for n in Ns]
        # curr_rdm_eigvals = [[None for bi in range(num_of_sampled_blocks)] for n in Ns]
        for ni in range(len(Ns)):
            n = Ns[ni]
            for bi in range(num_of_sampled_blocks):
                b = sampled_blocks[ni][bi]
                print(n, b)
                p2s[ni][bi] = get_block_purity(boundary_filename, n, b)
                p1s[ni][bi] = get_block_probability(boundary_filename, n, b)

        pickle.dump([tau_eigenvals, wilson_area, wilson_perimeter, p2s, p1s, sampled_blocks],
                    open(filename, 'wb'))


def analyze_normalized_p2_data(model, params, Ns, dirname, param_name, plot=True):
    import matplotlib.pyplot as plt
    wilson_areas = np.zeros(len(params))
    wilson_perimeters = np.zeros(len(params))
    tau_purities = np.zeros((len(params), 2))
    num_of_sampled_blocks = 63
    normalized_p2s = np.zeros((len(params), len(Ns), num_of_sampled_blocks))
    full_p2s = np.zeros(len(params))
    for pi in range(len(params)):
        param = params[pi]
        print(param)
        full_p2s[pi] = get_purity(2, 2, dirname, model, param_name, param)
        print(full_p2s[pi])
        filename = results_filename(dirname, model, param_name, param, Ns)
        [tau_eigenvals, wilson_area, wilson_perimeter, p2s, p1s, sampled_blocks] = pickle.load(open(filename, 'rb'))
        print(tau_eigenvals / np.sum(tau_eigenvals))
        tau_purities[pi, 0] = sum(np.abs(tau_eigenvals[:2] / sum(tau_eigenvals))**2)
        tau_purities[pi, 1] = sum(np.real(tau_eigenvals[2:])**2)
        for ni in range(len(Ns)):
            for bi in range(num_of_sampled_blocks):
                normalized_p2s[pi, ni, bi] = p2s[ni][bi] / p1s[ni][bi]**2

            wilson_areas[pi] = wilson_area
            wilson_perimeters[pi] = wilson_perimeter

    ff, axs = plt.subplots(2)
    for i in range(4):
        axs[0].plot(params, wilson_areas)
        axs[0].plot(params, wilson_perimeters)
    axs[0].legend([r'area law $\chi^2$'])
    axs[0].legend([r'perimeter law $\chi^2$'])
    axs[1].plot(params, full_p2s)
    plt.show()
    for sbi in range(len(sampled_blocks)):
        plt.pcolormesh(Ns, params, normalized_p2s[:, :, sbi])
        plt.title(sampled_blocks[sbi])
        plt.show()


model = sys.argv[1]
edge_dim = 2
if model == 'zohar':
    params = [[alpha, beta, gamma, delta] for alpha in [np.round(0.5 * i, 8) for i in range(-2, 3)] \
              for beta in [np.round(0.5 * i, 8) for i in range(-2, 3)] \
              for gamma in [np.round(0.5 * i, 8) for i in range(-2, 3)] \
              for delta in [np.round(0.5 * i, 8) for i in range(-2, 3)]]
    param_name = 'parmas'
elif model == 'zohar_deltas':
    params = [np.round(0.2 * a, 8) for a in range(-10, 11)]
    param_name = 'delta'
elif model == 'zohar_deltas_large_alpha':
    params = [np.round(0.2 * a, 8) for a in range(-10, 11)]
    param_name = 'delta'
elif model == 'zeros_diff':
    params = [np.round(0.1 * a, 8) for a in range(-9, 21)]
    param_name = 'alpha'
elif model == 'zohar_alpha':
    params = [np.round(0.2 * a, 8) for a in range(-10, 11)]
    param_name = 'alpha'
elif model == 'zohar_gamma':
    params = [np.round(0.2 * a, 8) for a in range(-10, 11)]
    param_name = 'gamma'
elif model == 'orus':
    params = [np.round(0.1 * i, 8) for i in range(3, 13)]
    param_name = 'g'
elif model == 'toric_c':
    params = [np.round(0.25 * i, 3) for i in range(8)]
    param_name = 'c'
    edge_dim = 4
dirname = 'results/gauge/' + model

charges = [1, 1, 1, 1]
bs = [0, 2 ** 20 - 1, 173524]
ns = [4] #, 8] # [12, 24, 48]
corner_nums = [1, 2] # [1, 2, 3, 4, 6]
gap_ls = [2, 4, 8] # [4, 10, 20]
corner_charges = [0, 1]

block_results = np.zeros((len(params), len(corner_nums), len(ns), len(gap_ls)))
for pi in range(len(params)):
    param = params[pi]
    filename = boundary_filname(dirname, model, param_name, param)
    for cni in range(len(corner_nums)):
        corner_num = corner_nums[cni]
        for ni in range(len(ns)):
            n = ns[ni]
            for gi in range(len(gap_ls)):
                gap_l = gap_ls[gi]
                res_filename = 'results/gauge/' + model + '/block_division_' + param_name + '_' + str(param) + '_n_' + str(n) + '_cnum_' + str(corner_num) + '_gap_' + str(gap_l)
                if not os.path.exists(res_filename):
                    pickle.dump(block_division_contribution(boundary_filname(dirname, model, param_name, param), n, corner_num, gap_l),
                                open(res_filename, 'wb'))
                block_results[pi, cni, ni, gi] = pickle.load(open(res_filename, 'rb'))


bs = [0, 2 ** 20 - 1, 173524]
ns = [4, 8] # [12, 24, 48]
corner_nums = [1, 2] # [1, 2, 3, 4, 6]
gap_ls = [2] # [4, 10, 20]
results = np.zeros((len(params), len(corner_nums), len(ns), len(bs), len(gap_ls), len(corner_charges)))
for pi in range(len(params)):
    param = params[pi]
    filename = boundary_filname(dirname, model, param_name, param)
    for ni in range(len(ns)):
        n = ns[ni]
        for gi in range(len(gap_ls)):
            gap_l = gap_ls[gi]
            for bi in range(len(bs)):
                b = bs[bi]
                for ci in range(len(corner_charges)):
                    c = corner_charges[ci]
                    for cni in range(len(corner_nums)):
                        corner_num = corner_nums[cni]
                        result_filename = 'results/gauge/' + model + '/circle_' + param_name + '_' + str(param) + '_n_' + str(n) + '_cnum_' + str(corner_num) \
                                        + '_b_' + str(b) + '_gap_l_' + str(gap_l) + '_c_' + str(c)
                        if not os.path.exists(result_filename):
                            p1 = get_block_probability(filename, n, b, corner_num=corner_num, corner_charges_i=c, gap_l=gap_l, edge_dim=edge_dim)
                            p2 = get_block_probability(filename, n, b, corner_num=corner_num, corner_charges_i=c, gap_l=gap_l, edge_dim=edge_dim, purity_mode=True)
                            print(param, corner_num, n, gap_l, p2, p1, p2 / p1**2)
                            n_p2 = p2 / p1**2
                            pickle.dump(n_p2, open(result_filename, 'wb'))
                        else:
                            n_p2 = pickle.load(open(result_filename, 'rb'))
                        results[pi, cni, ni, bi, gi, ci] = n_p2

dbg = 1