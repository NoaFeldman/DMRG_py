import numpy as np
import tensornetwork as tn
import basicOperations as bops
import pepsExpect as pe
import PEPS as peps
import pickle
import randomUs as ru
from typing import List
import os
import sys
import basicAnalysis as bans
import gc

X = np.array([[0, 1], [1, 0]])
I = np.eye(2)
d=2

def boundary_binary_string(i, N):
    curr = bin(i).split('b')[1]
    curr = '0' * (N - len(curr)) + curr
    last = np.prod([int(c) * 2 - 1 for c in curr])
    curr = str(int(last + 1 / 2)) + curr
    return curr


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


# TODO doubt everything above this line

def get_boundaries_from_file(filename, w, h):
    with open(filename, 'rb') as f:
        [upRow, downRow, leftRow, rightRow, openA, openB, A, B] = pickle.load(f)
        [upRow, downRow, leftRow, rightRow, te] = shrink_boundaries(upRow, downRow, leftRow, rightRow, bond_dim=4)
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
                    (np.sum(boundary[:2 * (ni + 1)]) + choice[ni]) % d]
    result += [boundary[n * 2 - 1], boundary[-1],
                (boundary[-1] + choice[-1] + boundary[n * 2 - 1]) % d, np.sum(boundary) % d]
    return result

# TODO here only d = 2
def get_2_by_n_explicit_block(filename, n, bi, d=2):
    [cUp, dUp, cDown, dDown, leftRow, rightRow, openA, openB, A, B] = get_boundaries_from_file(filename, w=n, h=2)
    # print([node.tensor.shape for node in [cUp, dUp, cDown, dDown, leftRow, rightRow]])
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


# Fig 5 in https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.3.033179
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
    elif model == 'orus':
        for i in range(d):
            for j in range(d):
                tensor[i, j, :, :, :, :] *= (1 + param)**(i + j)
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
        return A
    A = tn.Node(tensor.reshape([d] * 4 + [d**2]))
    return A

# A = toric_tensors_lgt_approach('toric_c', 0.5)
# T = bops.permute(bops.contract(A, A, '4', '4*'), [0, 2, 4, 6, 1, 3, 5, 7])
# singlet_projector_tensor = np.zeros((4, 4, 4**2))
# for i in [0, 3]:
#     for j in [0, 3]:
#         singlet_projector_tensor[i, j, i * 4 + j] = 1
# for i in [1, 2]:
#     for j in [1, 2]:
#         singlet_projector_tensor[i, j, i * 4 + j] = 1
# singlet_proj = tn.Node(singlet_projector_tensor)
# tau = bops.contract(bops.contract(bops.contract(bops.contract(
#     T, singlet_proj, '01', '01'), singlet_proj, '01', '01'), singlet_proj, '01', '01'), singlet_proj, '01', '01')
# tau_mat = tau.tensor.transpose([0, 2, 1, 3]).reshape([4**4, 4**4])
# dbg = 1

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
    bond_dim =  A[0].dimension
    projector_tensor = np.zeros([bond_dim] * 3)
    for i in range(d):
        projector_tensor[i, i, i] = 1
    projector = tn.Node(projector_tensor)
    tau = bops.contract(bops.contract(bops.contract(bops.contract(
        E0, projector, '01', '01'), projector, '01', '01'), projector, '01', '01'), projector, '01', '01')
    openA = tn.Node(np.kron(A.tensor, A.tensor.conj())\
                    .reshape([bond_dim**2] * 4 + [d**2] * 2).transpose([4, 0, 1, 2, 3, 5]))
    return A, tau, openA, projector


def results_filname(dirname, model, param_name, param, Ns):
    return dirname + '/normalized_p2_results_' + model + '_' + param_name + '_' + str(param) + '_Ns_' + str(Ns)
def boundary_filname(dirname, model, param_name, param):
    return dirname + '/toricBoundaries_gauge_' + model + '_' + param_name + '_' + str(param)


def shrink_boundaries(upRow, downRow, leftRow, rightRow, bond_dim):
    max_te = 0
    [upRow, leftRow, te] = bops.svdTruncation(bops.contract(upRow, leftRow, '3', '0'), [0, 1, 2], [3, 4, 5], '>>',
                                              maxBondDim=bond_dim, minBondDim=bond_dim)
    if len(te) > 0 and max(te) > max_te: max_te = max(te)
    [leftRow, downRow, te] = bops.svdTruncation(bops.contract(leftRow, downRow, '3', '0'), [0, 1, 2], [3, 4, 5], '>>',
                                                maxBondDim=bond_dim, minBondDim=bond_dim)
    if len(te) > 0 and max(te) > max_te: max_te = max(te)
    [downRow, rightRow, te] = bops.svdTruncation(bops.contract(downRow, rightRow, '3', '0'), [0, 1, 2], [3, 4, 5], '>>',
                                                 maxBondDim=bond_dim, minBondDim=bond_dim)
    if len(te) > 0 and max(te) > max_te: max_te = max(te)
    [rightRow, upRow, te] = bops.svdTruncation(bops.contract(rightRow, upRow, '3', '0'), [0, 1, 2], [3, 4, 5], '>>',
                                               maxBondDim=bond_dim, minBondDim=bond_dim)
    if len(te) > 0 and max(te) > max_te: max_te = max(te)
    return upRow, downRow, rightRow, leftRow, max_te


# TODO handle A != B
def get_boundaries(dirname, model, param_name, param, max_allowed_te=1e-10):
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
        upRow, downRow, rightRow, leftRow, max_te = shrink_boundaries(upRow, downRow, rightRow, leftRow, bond_dim)
        if max_te > max_allowed_te:
            bond_dim += 1
        else:
            break
    print('truncation error: ' + str(max_te) + ', bond dim: ' + str(bond_dim))
    [cUp, dUp, te] = bops.svdTruncation(upRow, [0, 1], [2, 3], '>>', maxBondDim=upRow.tensor.shape[0])
    [cDown, dDown, te] = bops.svdTruncation(downRow, [0, 1], [2, 3], '>>', maxBondDim=downRow.tensor.shape[0])
    return cUp, dUp, cDown, dDown, leftRow, rightRow, openA, openA

def get_full_purity(w, h, dirname, model, param_name, param):
    cUp, dUp, cDown, dDown, leftRow, rightRow, openA, openB = get_boundaries(dirname, model, param_name, param)
    norm = pe.applyLocalOperators(cUp, dUp, cDown, dDown, leftRow, rightRow, openA, openB, w, h,
                                  [tn.Node(np.eye(4)) for i in range(w * h)])
    leftRow.tensor /= norm**(h / 2)
    cUp = tn.Node(np.kron(cUp.tensor, cUp.tensor))
    dUp = tn.Node(np.kron(dUp.tensor, dUp.tensor))
    cDown = tn.Node(np.kron(cDown.tensor, cDown.tensor))
    dDown = tn.Node(np.kron(dDown.tensor, dDown.tensor))
    leftRow = tn.Node(np.kron(leftRow.tensor, leftRow.tensor))
    rightRow = tn.Node(np.kron(rightRow.tensor, rightRow.tensor))
    A = tn.Node(np.kron(openA.tensor, openA.tensor))
    B = tn.Node(np.kron(openB.tensor, openB.tensor))
    single_swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    double_swap = np.kron(single_swap, single_swap).reshape([2] * 8).transpose([3, 2, 1, 0, 4, 5, 6, 7]).reshape(
        [2 ** 4, 2 ** 4])
    full_purity = pe.applyLocalOperators(cUp, dUp, cDown, dDown, leftRow, rightRow, A, B, h, w,
                                         [tn.Node(double_swap) for i in range(w * h)])
    return full_purity


def wilson_expectations(model, param, param_name, dirname, plot=False, d=2):
    cUp, dUp, cDown, dDown, leftRow, rightRow, openA, openA = get_boundaries(dirname, model, param_name, param)

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
        filename = results_filname(dirname, model, param_name, param, Ns)
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
        normalized_p2s = [np.zeros(num_of_sampled_blocks) for n in Ns]
        # curr_rdm_eigvals = [[None for bi in range(num_of_sampled_blocks)] for n in Ns]
        for ni in range(len(Ns)):
            n = Ns[ni]
            for bi in range(num_of_sampled_blocks):
                b = sampled_blocks[ni][bi]
                print(n, b)
                block = get_2_by_n_explicit_block(boundary_filename, n, b, d=d)
                normalized_p2s[ni][bi] = np.real(np.matmul(block, block).trace()) / block.trace()**2
                rdm_eigvals = np.linalg.eigvals(block)
                # curr_rdm_eigvals[ni][bi] = rdm_eigvals
                if min(rdm_eigvals) < -1e-12:
                    print('Negative rho eigenvalue!!!')
                    print(np.round(rdm_eigvals, 20))

        pickle.dump([tau_eigenvals, wilson_area, wilson_perimeter, normalized_p2s, sampled_blocks], open(filename, 'wb'))

def analyze_normalized_p2_data(model, params, Ns, dirname, param_name):
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
        full_p2s[pi] = get_full_purity(2, 2, dirname, model, param_name, param)
        print(full_p2s[pi])
        filename = results_filname(dirname, model, param_name, param, Ns)
        [tau_eigenvals, wilson_area, wilson_perimeter, curr_normalized_p2s, sampled_blocks] = pickle.load(open(filename, 'rb'))
        print(tau_eigenvals / np.sum(tau_eigenvals))
        tau_purities[pi, 0] = sum(np.abs(tau_eigenvals[:2] / sum(tau_eigenvals))**2)
        tau_purities[pi, 1] = sum(np.real(tau_eigenvals[2:])**2)
        for ni in range(len(Ns)):
            for bi in range(num_of_sampled_blocks):
                normalized_p2s[pi, ni, bi] = curr_normalized_p2s[ni][bi]

        wilson_areas[pi] = wilson_area
        wilson_perimeters[pi] = wilson_perimeter

    ff, axs = plt.subplots(3)
    for i in range(4):
        axs[0].plot(params, wilson_areas)
        axs[0].plot(params, wilson_perimeters)
    axs[0].legend([r'area law $\chi^2$'])
    axs[0].legend([r'perimeter law $\chi^2$'])
    axs[1].plot(params, full_p2s)
    # axs[2].plot(params, tau_purities[:, 0])
    # axs[2].plot(params, tau_purities[:, 1], '--')
    for ni in range(len(Ns)):
        for bi in range(num_of_sampled_blocks):
            axs[2].plot(params, normalized_p2s[:, ni, bi])
    plt.show()

cs = [0.5, 1.0]
dirname = 'results/gauge/toric_c'
param_name = 'c'
Ns = [2, 4]
model = 'toric_c'
normalized_p2s_data(model, cs, Ns, dirname, param_name)
analyze_normalized_p2_data(model, cs, Ns, dirname, param_name)