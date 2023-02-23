import matplotlib.pyplot as plt

import basicOperations as bops
import numpy as np
import DMRG as dmrg
import pickle
import basicDefs
import tensornetwork as tn
import sys
import PEPS as peps
import scipy.linalg as linalg
import os
import functools as ft

# https://journals.aps.org/prb/pdf/10.1103/PhysRevB.91.115137

def prepare_g_tensor(ising_lambda, J, delta, model='transverse'):
    if model == 'transverse':
        single_op = tn.Node(linalg.expm(-Z * ising_lambda * delta / 2).T)
    elif model == 't_gate':
        single_op = tn.Node(linalg.expm(-np.diag([1, np.exp(1j * np.pi / 4)]) * ising_lambda * delta / 2).T)
    pair_op = tn.Node(linalg.expm(-delta * np.kron(J * X, X)).reshape([2] * 4).transpose([2, 0, 3, 1]))
    l, s, r, te = bops.svdTruncation(pair_op, [0, 1], [2, 3], '>*<')
    l = bops.contract(l, tn.Node(np.sqrt(s.tensor)), '2', '0')
    r = bops.contract(tn.Node(np.sqrt(s.tensor)), r, '1', '0')
    return bops.permute(bops.contract(bops.contract(
        single_op, bops.permute(bops.contract(r, l, '1', '1'), [2, 1, 0, 3]), '1', '0'), single_op, '1', '0'),
        [0, 3, 1, 2])


def get_environment(node, g):
    t_matrix = bops.contract(bops.contract(node, g, '1', '0'), node, '2', '1*').tensor.transpose([0, 2, 4, 1, 3, 5])\
        .reshape([node[0].dimension**2 * g[2].dimension] * 2)
    vals_r, vecs_r = np.linalg.eig(t_matrix)
    inds_max = np.where(vals_r - np.max(vals_r) == 0)[0]
    HR_tensor = vecs_r[:, inds_max[0]].reshape([node[0].dimension, g[2].dimension, node[0].dimension])
    for i in range(1, len(inds_max)):
        HR_tensor += vecs_r[:, inds_max[i]].reshape([node[0].dimension, g[2].dimension, node[0].dimension])
    HR = tn.Node(HR_tensor)
    vals_l, vecs_l = np.linalg.eig(t_matrix.T)
    inds_max = np.where(vals_l - np.max(vals_l) == 0)[0]
    HL_tensor = vecs_l[:, inds_max[0]].reshape([node[0].dimension, g[2].dimension, node[0].dimension])
    for i in range(1, len(inds_max)):
        HL_tensor += vecs_l[:, inds_max[i]].reshape([node[0].dimension, g[2].dimension, node[0].dimension])
    HL = tn.Node(HL_tensor)
    return HR, HL


def cannonical_node_step(lambda_r, Gamma):
    L = bops.contract(bops.contract(lambda_r, Gamma, '1', '0'), bops.contract(lambda_r, Gamma, '1', '0'), '1', '1*')\
        .tensor.transpose([1, 3, 0, 2]).reshape([lambda_r[0].dimension**2] * 2)
    vals, vecs = np.linalg.eig(L)
    max_inds = np.where(np.round(np.abs(vals) - max(np.abs(vals)), 8) == 0)[0]
    VLT_tensor = vecs[:, max_inds[0]].reshape([lambda_r[0].dimension] * 2)
    for i in range(1, len(max_inds)):
        VLT_tensor += vecs[:, max_inds[i]].reshape([lambda_r[0].dimension] * 2)
    up_l, s_l, down_l, te = bops.svdTruncation(tn.Node(VLT_tensor), [0], [1], '>*<', maxTrunc=0)
    YT = bops.contract(tn.Node(np.diag([i if i > 1e-4 else 0 for i in np.sqrt(np.diag(s_l.tensor))])), up_l, '0', '1')
    YT_inverse = tn.Node(np.matmul(up_l.tensor.conj(), np.diag([1/i if i > 1e-4 else 0 for i in np.sqrt(np.diag(s_l.tensor))])))
    R = bops.contract(bops.contract(Gamma, lambda_r, '2', '0'), bops.contract(Gamma, lambda_r, '2', '0'), '1', '1*')\
        .tensor.transpose([0, 2, 1, 3]).reshape([lambda_r[0].dimension**2] * 2)
    vals, vecs = np.linalg.eig(R)
    max_inds = np.where(np.round(np.abs(vals) - max(np.abs(vals)), 8) == 0)[0]
    VR_tensor = vecs[:, max_inds[0]].reshape([lambda_r[0].dimension] * 2)
    for i in range(1, len(max_inds)):
        VR_tensor += vecs[:, max_inds[i]].reshape([lambda_r[0].dimension] * 2)
    up_r, s_r, down_r, te = bops.svdTruncation(tn.Node(VR_tensor), [0], [1], '>*<', maxTrunc=0)
    X = bops.contract(up_r, tn.Node(np.diag([i if i > 1e-4 else 0 for i in np.sqrt(np.diag(s_r.tensor))])), '1', '0')
    X_inverse = tn.Node(np.matmul(np.diag([1/i if i > 1e-4 else 0 for i in np.sqrt(np.diag(s_r.tensor))]), up_r.tensor.conj().T))
    row = bops.contract(YT, bops.contract(lambda_r, X, '1', '0'), '1', '0')
    U1, lambda_tilde, V1, te = bops.svdTruncation(row, [0], [1], '>*<', normalize=True)
    Gamma_tilde = bops.contract(bops.contract(bops.contract(bops.contract(
        V1, X_inverse, '1', '0'), Gamma, '1', '0'), YT_inverse, '2', '0'), U1, '2', '0')
    return lambda_tilde, Gamma_tilde


def cannonical_node(node):
    stop = False
    lambda_r, Gamma, te, = bops.svdTruncation(node, [0], [1, 2], '<<', normalize=True)
    if lambda_r[0].dimension > lambda_r[1].dimension:
        Gamma_tensor = np.zeros((lambda_r[0].dimension, Gamma[1].dimension, Gamma[2].dimension), dtype=complex)
        Gamma_tensor[:lambda_r[1].dimension, :, :] = Gamma.tensor
        Gamma.tensor = Gamma_tensor
        lambda_r_tensor = np.zeros((lambda_r[0].dimension, lambda_r[0].dimension), dtype=complex)
        lambda_r_tensor[:, :lambda_r[1].dimension] = lambda_r.tensor
        lambda_r.tensor = lambda_r_tensor
    while not stop:
        lambda_r, Gamma = cannonical_node_step(lambda_r, Gamma)
        is_id_r = bops.contract(bops.contract(lambda_r, Gamma, '1', '0'),
                                bops.contract(lambda_r, Gamma, '1', '0'), '01', '01*').tensor
        Gamma.tensor /= np.sqrt(is_id_r[0, 0])
        is_id_l = bops.contract(bops.contract(Gamma, lambda_r, '2', '0'),
                                bops.contract(Gamma, lambda_r, '2', '0'), '12', '12*').tensor
        if np.all(np.round(is_id_r / is_id_r[0, 0], 5) == np.eye(len(is_id_r))) and np.all(np.round(is_id_l, 5) == np.eye(len(is_id_l))):
            stop = True
    return lambda_r, Gamma


def iTEBD_step(lambda_r, Gamma, g, chi):
    node = bops.contract(lambda_r, Gamma, '1', '0')
    node = tn.Node(bops.contract(node, g, '1', '0').tensor.transpose([0, 3, 2, 1, 4])\
        .reshape([node[0].dimension * g[2].dimension, g[1].dimension, node[0].dimension * g[2].dimension]))
    lambda_r, Gamma = cannonical_node(node)
    lambda_r.tensor = lambda_r.tensor[:chi, :chi]
    Gamma.tensor = Gamma.tensor[:chi, :, :chi]
    return lambda_r, Gamma


def converged(lambda_1, Gamma_1, lambda_2, Gamma_2, accuracy=1e-6):
    rho_1 = bops.contract(bops.contract(bops.contract(lambda_1, Gamma_1, '1', '0'), lambda_1, '2', '0'),
                          bops.contract(bops.contract(lambda_1, Gamma_1, '1', '0'), lambda_1, '2', '0'), '02', '02*')\
        .tensor
    rho_1 /= rho_1.trace()
    rho_2 = bops.contract(bops.contract(bops.contract(lambda_2, Gamma_2, '1', '0'), lambda_2, '2', '0'),
                          bops.contract(bops.contract(lambda_2, Gamma_2, '1', '0'), lambda_2, '2', '0'), '02', '02*')\
        .tensor
    rho_2 /= rho_2.trace()
    fidelity = linalg.sqrtm(np.matmul(linalg.sqrtm(rho_1), np.matmul(rho_2, linalg.sqrtm(rho_1)))).trace()**2
    if 1 - fidelity > accuracy:
        return False
    return True


initial_c_tensor = np.zeros((4, 2, 4), dtype=complex)
initial_c_tensor[0, 0, 0] = 1
initial_c_tensor[1, 1, 1] = 1
initial_c_tensor[2, 1, 3] = 1 / np.sqrt(2)
initial_c_tensor[3, 1, 3] = 1 / np.sqrt(2)
initial_c_tensor[3, 0, 2] = 1 / np.sqrt(2)
initial_c_tensor[0, 1, 2] = 1 / np.sqrt(2)
initial_c_tensor[3, 1, 2] = 1 / np.sqrt(2)
initial_c_tensor[2, 0, 3] = 1 / np.sqrt(2)
initial_c_tensor[2, 1, 0] = 1 / np.sqrt(2)
initial_c = tn.Node(initial_c_tensor)
l, lambda_c, r, te = bops.svdTruncation(initial_c, [0], [1, 2], '>*<', normalize=True)
initial_c = bops.contract(lambda_c, r, '1', '0')
cannonical_node(initial_c)

J = -1
lambda_step = 0.1
lambda_critical_step = 0.04
phase_transition = 1
ising_lambdas = [np.round(phase_transition + lambda_critical_step * i, 8) for i in range(-24, 0)] + \
                [np.round(phase_transition + lambda_critical_step * i, 8) for i in range(24, 0, -1)] # + \
                # [np.round(phase_transition + lambda_critical_step * i / 10, 8) for i in range(-10, 0)] \
                # +\
                # [np.round(phase_transition + lambda_critical_step * i / 10, 8) for i in range(10, 0, -1)]
ising_lambdas.sort()
I = np.eye(2, dtype=complex)
Z = np.diag([1, -1])
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]])

angle_step = 30
thetas = [np.round(i/angle_step, 3) for i in range(angle_step)]
phis = [np.round(i/angle_step, 3) for i in range(angle_step)]
sigmas = [np.round(i/angle_step, 3) for i in range(angle_step)]
m2s = np.zeros(len(ising_lambdas))
m2s_mins = np.zeros(len(ising_lambdas))
min_thetas = np.zeros(len(ising_lambdas))
min_phis = np.zeros(len(ising_lambdas))
min_sigmas = np.zeros(len(ising_lambdas))
alpha_x = np.zeros(len(ising_lambdas))
alpha_y = np.zeros(len(ising_lambdas))
alpha_z = np.zeros(len(ising_lambdas))
alpha_abs = np.zeros(len(ising_lambdas))
magnetisations = np.zeros(len(ising_lambdas), dtype=complex)
nearest_neighbors = np.zeros(len(ising_lambdas), dtype=complex)
p2s = np.zeros(len(ising_lambdas))
lambda_0_tensor = np.zeros([2, 2, 2], dtype=complex)
epsilon = 0.01
lambda_0_tensor[0, 0, 0] = 1 # epsilon
lambda_0_tensor[0, 1, 0] = 1
lambda_0_tensor[1, 0, 1] = 1 # epsilon
lambda_0_tensor[1, 1, 1] = -1
lambda_0_tensor[1, 0, 0] = epsilon
lambda_0_tensor[1, 1, 0] = epsilon
lambda_0_tensor[0, 0, 1] = -epsilon
lambda_0_tensor[0, 1, 1] = epsilon
lambda_r, Gamma = tn.Node(np.diag([1, 1]) / np.sqrt(2)), tn.Node(lambda_0_tensor)

model = 'transverse'
for li in range(len(ising_lambdas)):
    non_normalized_m2s = np.zeros((len(thetas), len(phis), len(sigmas)))
    ising_lambda = ising_lambdas[li]
    print(ising_lambda)
    results_filename = 'results/magic/bmps_ising_lambda_' + str(ising_lambda) + '_J_' + str(J)
    if model != 'transverse':
        results_filename += '_' + model
    if os.path.exists(results_filename):
        data = pickle.load(open(results_filename, 'rb'))
        [lambda_r, Gamma, non_normalized_m2s] = data
    else:
        steps = 100000
        accuracy = 1e-3
        delta = 1e-2
        for i in range(steps):
            g = prepare_g_tensor(ising_lambda, J, delta=delta, model=model)
            lambda_r_new, Gamma_new = iTEBD_step(lambda_r, Gamma, g, chi=128)
            if converged(lambda_r, Gamma, lambda_r_new, Gamma_new, accuracy=delta**3):
                print(bops.contract(bops.contract(bops.contract(lambda_r, Gamma, '1', '0'), lambda_r, '2', '0'),
                                    bops.contract(bops.contract(lambda_r, Gamma, '1', '0'), lambda_r, '2', '0'), '02',
                                    '02*') \
                      .tensor)
                if delta <= accuracy:
                    break
                delta /= 10
            lambda_r, Gamma = lambda_r_new, Gamma_new
    non_normalized_m2s = np.zeros((len(thetas), len(phis), len(sigmas)))
    lambda_shrinked = tn.Node(lambda_r.tensor[:2, :2] / np.sqrt(sum(np.diag(lambda_r.tensor)[:2]**2)))
    Gamma_shrinked = tn.Node(Gamma.tensor[:2, :, :2])
    node = bops.contract(lambda_shrinked, Gamma_shrinked, '1', '0')
    node_4 = tn.Node(
        np.kron(node.tensor, np.kron(node.tensor.conj(), np.kron(node.tensor, node.tensor.conj()))))
    for ti in range(len(thetas)):
        theta = thetas[ti] * np.pi / 4
        for pi in range(len(phis)):
            phi = phis[pi] * np.pi / 4
            for si in range(len(sigmas)):
                sigma = sigmas[si] * np.pi / 4
                paulis = [ft.reduce(np.matmul,
                    [linalg.expm(1j * sigma * Y), linalg.expm(1j * theta * Z), linalg.expm(1j * phi * X), op,
                     linalg.expm(-1j * phi * X), linalg.expm(-1j * theta * Z), linalg.expm(-1j * sigma * Y)])
                    for op in [X, Y, Z]]
                op_tensor = np.kron(I, np.kron(I, np.kron(I, I)))
                op_4 = tn.Node(op_tensor)
                for pauli in paulis: op_tensor += np.kron(pauli, np.kron(pauli, np.kron(pauli, pauli)))
                magic_T = bops.contract(bops.contract(node_4, op_4, '1', '0'), node_4, '2', '1*').tensor.transpose([0, 2, 1, 3]) \
                    .reshape([node_4[0].dimension ** 2, node_4[2].dimension ** 2])
                vals = np.linalg.eigvals(magic_T)
                non_normalized_m2s[ti, pi, si] = np.log(np.amax(np.abs(vals)))
        pickle.dump([lambda_r, Gamma, non_normalized_m2s], open(results_filename, 'wb'))
    # for i in range(len(non_normalized_m2s)):
    #     plt.pcolormesh(non_normalized_m2s[i])
    #     plt.colorbar()
    #     plt.title(str(i) + ' ' + str(ising_lambda))
    #     plt.show()
    print(len(lambda_r.tensor), np.diag(lambda_r.tensor), np.sum(lambda_r.tensor[:2, :2] ** 2) / np.sum(lambda_r.tensor ** 2))
    print(bops.contract(bops.contract(bops.contract(lambda_r, Gamma, '1', '0'), lambda_r, '2', '0'),
                          bops.contract(bops.contract(lambda_r, Gamma, '1', '0'), lambda_r, '2', '0'), '02', '02*')\
        .tensor)
    m2s[li] = non_normalized_m2s[0, 0, 0]
    m2s_mins[li] = np.amax(non_normalized_m2s)
    min_thetas[li], min_phis[li], min_sigmas[li] = [np.where(non_normalized_m2s - m2s_mins[li] == 0)[i][0] for i in range(3)]
    # LambdaC.tensor /= np.sqrt(np.sum(LambdaC.tensor**2))
    p2s[li] = np.sum(np.abs(lambda_r.tensor**4))
    rho = bops.contract(bops.contract(bops.contract(lambda_r, Gamma, '1', '0'), lambda_r, '2', '0'),
                        bops.contract(bops.contract(lambda_r, Gamma, '1', '0'), lambda_r, '2', '0'), '02', '02*').tensor
    alpha_x[li] = np.matmul(rho, X).trace()
    alpha_y[li] = np.matmul(rho, Y).trace()
    alpha_z[li] = np.matmul(rho, Z).trace()
    alpha_abs[li] = alpha_x[li]**2 + alpha_y[li]**2 + alpha_z[li]**2
    magnetisations[li] = bops.contract(bops.contract(bops.contract(bops.contract(lambda_r, Gamma, '1', '0'), lambda_r, '2', '0'),
                                                     tn.Node(Z), '1', '0'),
                                       bops.contract(bops.contract(lambda_r, Gamma, '1', '0'), lambda_r, '2', '0'), '021', '012*').tensor
    pair = bops.contract(bops.contract(bops.contract(bops.contract(
        lambda_r, Gamma, '1', '0'), lambda_r, '2', '0'), Gamma, '2', '0'), lambda_r, '3', '0')
    nearest_neighbors[li] = bops.contract(bops.contract(bops.contract(
        pair, tn.Node(Z), '1', '0'), tn.Node(Z), '1', '0'), pair, '0231', '0123*').tensor
    dbg = 1
plt.plot(ising_lambdas, p2s)
# plt.plot(ising_lambdas, np.real(magnetisations))
# plt.plot(ising_lambdas, np.real(nearest_neighbors))
plt.plot(ising_lambdas, m2s / np.log(2))
plt.plot(ising_lambdas, m2s_mins / np.log(2), '--')
# plt.plot(ising_lambdas, min_thetas * 0.1)
# plt.plot(ising_lambdas, min_phis * 0.1, '--')
# plt.plot(ising_lambdas, min_sigmas * 0.1, ':')
plt.plot(ising_lambdas, alpha_x**2)
plt.plot(ising_lambdas, alpha_y**2)
plt.plot(ising_lambdas, alpha_z**2)
plt.plot(ising_lambdas, alpha_abs)
plt.legend([r'$p_2$', # r'$\langle \sigma^z\rangle$', r'$\langle \sigma^z\otimes\sigma^z\rangle$',
            r'$M_2$', r'$M_2 min$',
            r'$\alpha_x$', r'$\alpha_y$', r'$\alpha_z$', r'$\alpha^2$'])
plt.xlabel('h')
plt.show()