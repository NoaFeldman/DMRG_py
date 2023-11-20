import numpy as np
import tensornetwork as tn
import basicOperations as bops
import pepsExpect as pe
import PEPS as peps
import pickle
import os
import sys
import gc
from scipy.linalg import expm

# Everything here is based on https://journals.aps.org/prd/pdf/10.1103/PhysRevD.107.014505
d = 2
I = np.eye(d)
X = np.array([[0, 1], [1, 0]])
Z = np.diag([1, -1])
sqrt_X = np.array([[1 + 1j, 1 - 1j],
                   [1 - 1j, 1 + 1j]]) * 0.5
s_plus = np.array([[0, 0], [1, 0]])


def get_T(F, params):
    if F == 1:
        y, z = params
        return np.array([[0, -z, -1j * y, -1j * z],
                         [0, 0, -1j * z, y],
                         [0, 0, 0, z]])


def get_A(F, params):
    T = get_T(F, params)
    r_dagger = np.kron(np.kron(np.kron(s_plus, I), I), I)
    u_dagger = np.kron(np.kron(np.kron(I, s_plus), I), I)
    l_dagger = np.kron(np.kron(np.kron(I, I), s_plus), I)
    d_dagger = np.kron(np.kron(np.kron(I, I), I), s_plus)
    if F == 1:
        ops = [r_dagger, u_dagger, l_dagger, d_dagger]
    elif F == 2:
        ops = [np.kron(op, np.eye(d**4)) for op in [r_dagger, u_dagger, l_dagger, d_dagger]] + \
              [np.kron(np.eye(d**4), op) for op in [r_dagger, u_dagger, l_dagger, d_dagger]]
    A = np.eye((d * F)**4, dtype=complex)
    for row in range(len(T)):
        for col in range(len(T[0])):
            A += 2 * T[row, col] * np.matmul(ops[row], ops[col])
    return tn.Node(A.reshape([d * F] * 8))


def get_Ug(F):
    if F == 1:
        ug_tensor = np.zeros((d*F, d*F, d**2, d*F, d*F, d**2))
        for ri in range(d*F):
            for pri in range(d):
                for ui in range(d*F):
                    for pui in range(d):
                        ug_tensor[ri, ui, d*pri + pui, ri, ui, d*((pri + ri) % d) + (pui + ui) % d] = 1
        return tn.Node(ug_tensor)


def get_w_1(F):
    if F == 1:
        return tn.Node(np.eye(d*F))


def get_w_2(F):
    if F == 1:
        return tn.Node(np.diag([1, 1j]))


def w_loop_explicit(F, node):
    top_left = bops.contract(node, tn.Node(np.kron(X, I)), '4', '0')
    top_right = node
    bottom_left = bops.contract(node, tn.Node(np.kron(X, X)), '4', '0')
    bottom_right = bops.contract(node, tn.Node(np.kron(I, X)), '4', '0')
    top_left, top_right, bottom_left, bottom_right, rho = \
        [tn.Node(bops.contract(el, node, '4', '4').tensor.transpose([0, 4, 1, 5, 2, 6, 3, 7]).reshape([(d*F)**2] * 4)) \
         for el in [top_left, top_right, bottom_left, bottom_right, node]]
    exp_val = bops.contract(
        bops.contract(top_left, top_right, '02', '20'),
        bops.contract(bottom_left, bottom_right, '02', '20'), '0213', '1302').tensor
    norm = bops.contract(
        bops.contract(rho, rho, '02', '20'), bops.contract(rho, rho, '02', '20'), '0213', '1302').tensor
    return exp_val / norm


def q_coeff_exact(q_configuration, A, Omega, w1, w2):
    AOmega = bops.contract(A, Omega, '4567', '0123')
    uqs = [tn.Node(I), tn.Node(Z)]
    q = [int(c) for c in bin(q_configuration).split('b')[1].zfill(8)]
    top_left = bops.permute(bops.contract(bops.contract(uqs[q[0]], AOmega, '1', '0'), uqs[q[1]], '1', '1'), [0, 4, 1, 2, 3])
    top_right = bops.permute(bops.contract(bops.contract(uqs[q[2]], AOmega, '1', '0'), uqs[q[3]], '1', '1'), [0, 4, 1, 2, 3])
    bottom_left = bops.permute(bops.contract(bops.contract(uqs[q[4]], AOmega, '1', '0'), uqs[q[5]], '1', '1'), [0, 4, 1, 2, 3])
    bottom_right = bops.permute(bops.contract(bops.contract(uqs[q[6]], AOmega, '1', '0'), uqs[q[7]], '1', '1'), [0, 4, 1, 2, 3])
    [top_left, top_right, bottom_left, bottom_right] = \
        [bops.permute(bops.contract(bops.contract(
            w1, node, '0', '0'), w2, '1', '0'), [0, 4, 1, 2, 3]) for node in
            [top_left, top_right, bottom_left, bottom_right]]
    [top_left, top_right, bottom_left, bottom_right] = \
        [tn.Node(bops.contract(node, Omega, '4', '4').tensor.transpose([0, 4, 1, 5, 2, 6, 3, 7]).reshape([(d*F)**2] * 4))
         for node in [top_left, top_right, bottom_left, bottom_right]]
    return bops.contract(bops.contract(top_left, top_right, '02', '20'),
                         bops.contract(bottom_left, bottom_right, '02', '20'), '0213', '1302').tensor * 1


def cov_mat_from_node(node: tn.Node, F: int):
    gammas = [tn.Node(s_plus + s_plus.T), tn.Node(1j * (s_plus - s_plus.T))]
    cov_exact = np.zeros((8 * F, 8 * F), dtype=complex)
    for legi in range(4 * F):
        for gi in range(2):
            for legj in range(4 * F):
                for gj in range(2):
                    # TODO F>1 is not really handled here at all
                    cov_exact[legi + gi * (4 * F), legj + gj * (4 * F)] = \
                        1j * int(not (legi == legj and gi == gj)) * \
                        (int(legi + gi * (4 * F) > legj + gj * (4 * F)) * 2 - 1) * \
                        bops.contract(node,
                            bops.permute(bops.contract(bops.permute(bops.contract(
                                node, gammas[gi], [legi], '0'),
                                list(range(legi)) + [4] + list(range(legi, 4))),
                                gammas[gj], [legj], '0'),
                                list(range(legj)) + [4] + list(range(legj, 4))),
                                '01234*', '01234').tensor * 1
    cov_exact /= bops.contract(node, node, '01234*', '01234').tensor
    return cov_exact


def norm_of_q_config(T, A, Omega, w1, w2, q_configuration, N):
    AOmega = bops.contract(A, Omega, '4567', '0123')
    uqs = [tn.Node(I), tn.Node(Z)]
    q = [int(c) for c in bin(q_configuration).split('b')[1].zfill(8)]
    top_left = bops.permute(bops.contract(bops.contract(uqs[q[0]], AOmega, '1', '0'), uqs[q[1]], '1', '1'), [0, 4, 1, 2, 3])
    top_right = bops.permute(bops.contract(bops.contract(uqs[q[2]], AOmega, '1', '0'), uqs[q[3]], '1', '1'), [0, 4, 1, 2, 3])
    bottom_left = bops.permute(bops.contract(bops.contract(uqs[q[4]], AOmega, '1', '0'), uqs[q[5]], '1', '1'), [0, 4, 1, 2, 3])
    bottom_right = bops.permute(bops.contract(bops.contract(uqs[q[6]], AOmega, '1', '0'), uqs[q[7]], '1', '1'), [0, 4, 1, 2, 3])
    [top_left, top_right, bottom_left, bottom_right] = \
        [bops.permute(bops.contract(bops.contract(
            w1, node, '0', '0'), w2, '1', '0'), [0, 4, 1, 2, 3]) for node in
            [top_left, top_right, bottom_left, bottom_right]]
    cov_R_exact = cov_mat_from_node(Omega, F)
    [tl, tr, bl, br] = [cov_mat_from_node(node, F) for node in [top_left, top_right, bottom_left, bottom_right]]
    print([np.round(np.linalg.det(np.eye(len(tl), dtype=complex) - np.matmul(mat, cov_R_exact)), 8) for mat in [tl, tr, bl, br]])
    return np.prod([np.linalg.det(np.eye(len(tl), dtype=complex) - np.matmul(mat, cov_R_exact)) for mat in [tl, tr, bl, br]])
    #
    # uqs = [tn.Node(I), tn.Node(Z)]
    # q = [int(c) for c in bin(q_configuration).split('b')[1].zfill(N)]
    # wOmega = bops.permute(bops.contract(bops.contract(w1, AOmega, '1*', '0'), w2, '1', '1*'), [0, 4, 1, 2, 3])
    # psi_L_qs = [None for ni in range(N//2)]
    # for qi in range(N//2):
    #     psi_L_curr_node = bops.permute(bops.contract(bops.contract(
    #         uqs[q[0]], wOmega, '1', '0'), uqs[q[1]], '1', '1'), [0, 4, 1, 2, 3])
    #     cov_curr_L = cov_mat_from_node(bops.contract(psi_L_curr_node, tn.Node(np.kron(H, H)), '4', '0'), F)
    #     psi_L_qs[qi] = cov_curr_L
    # return np.sqrt(np.prod([np.linalg.det(
    #     np.eye(len(cov_R_exact), dtype=complex) - np.matmul(Gamma, cov_R_exact)) for Gamma in psi_L_qs]) / 2)


F = 1
Omega_tensor = np.zeros((d*F, d*F, d*F, d*F, d**2))
Omega_tensor[0, 0, 0, 0, 0] = 1
Omega = tn.Node(Omega_tensor)
A = get_A(F, [1, 1])
AOmega = bops.contract(A, Omega, '4567', '0123')
Ug = get_Ug(F)
UAOmega = bops.permute(bops.contract(Ug, AOmega, '012', '014'), [0, 1, 3, 4, 2])
w1 = get_w_1(F)
w2 = get_w_2(F)
node = bops.permute(bops.contract(bops.contract(
    w1, UAOmega, '0', '0'), w2, '1', '0'), [0, 4, 1, 2, 3])
W = w_loop_explicit(F, node)
vec = bops.contract(bops.contract(node, node, '02', '20'),
                    bops.contract(node, node, '02', '20'),
                    '0314', '1403').tensor.reshape([4**4])
H = np.array([[1, 1], [1, -1]]) * np.sqrt(0.5)
node_x_basis = bops.contract(node, tn.Node(np.kron(H, H)), '4', '0')
vec_x_basis = bops.contract(bops.contract(node_x_basis, node_x_basis, '02', '20'),
                    bops.contract(node_x_basis, node_x_basis, '02', '20'),
                    '0314', '1403').tensor.reshape([4**4])
tst = np.array([q_coeff_exact(q, A, Omega, w1, w2) for q in range(256)])
norm_qs = [norm_of_q_config(get_T(F, [1, 1]), A, Omega, w1, w2, q, 8) for q in np.where(vec)[0]]
dbg = 1