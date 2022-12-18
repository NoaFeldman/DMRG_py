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

J = 1
lambda_step = 0.1
lambda_critical_step = 0.04
phase_transition = 1
ising_lambdas = [np.round(phase_transition + lambda_critical_step * i, 8) for i in range(-9, 13)]
I = np.eye(2, dtype=complex)
Z = np.diag([1, -1])
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]])

angle_step = 10
thetas = [np.round(0.1 * i, 3) for i in range(angle_step)]
phis = [np.round(0.1 * i, 3) for i in range(angle_step)]
m2s = np.zeros(len(ising_lambdas))
m2s_mins = np.zeros(len(ising_lambdas))
p2s_C = np.zeros(len(ising_lambdas))
p2s_D = np.zeros(len(ising_lambdas))
for li in range(len(ising_lambdas)):
    non_normalized_m2s = np.zeros((len(thetas), len(phis)))
    ising_lambda = ising_lambdas[li]
    print(ising_lambda)
    results_filename = 'magic/results/bmps_ising_lambda_' + str(ising_lambda) + '_J_' + str(J)
    if os.path.exists(results_filename):
        data = pickle.load(open(results_filename, 'rb'))
        [GammaC, LambdaC, GammaD, LambdaD, non_normalized_m2s] = data
    else:
        dbeta = 1e-4
        H_term = J * np.kron(Z, Z) + ising_lambda * np.kron(X, np.eye(2))
        imag_time_term = linalg.expm(-dbeta * H_term).reshape([1, 2, 2, 2, 2, 1]).transpose([0, 1, 3, 2, 4, 5])
        A, B, te = bops.svdTruncation(tn.Node(imag_time_term), [0, 1, 2], [3, 4, 5], '>>')
        A = tn.Node(A.tensor.transpose([1, 3, 2, 0]))
        B = tn.Node(B.tensor.transpose([1, 3, 2, 0]))
        initial_c_tensor = np.zeros((2, 2, 2), dtype=complex)
        initial_c_tensor[0, 0, 0] = 1
        initial_c_tensor[1, 1, 1] = 1
        initial_c = tn.Node(initial_c_tensor)
        initial_lambda = tn.Node(np.array([1, 1], dtype=complex))

        GammaC, LambdaC, GammaD, LambdaD = peps.getBMPSRowOps(initial_c, initial_lambda, initial_c, initial_lambda, A, B, int(1e6), 128)
        LambdaC_shrinked = tn.Node(LambdaC.tensor[:4] / np.sqrt(np.sum(np.abs(LambdaC.tensor[:4])**2)))
        print(np.sum(LambdaC_shrinked.tensor**2) / (np.sum(LambdaC.tensor**2) / np.sum(np.abs(LambdaC.tensor[:4])**2)))
        LambdaD_shrinked = tn.Node(LambdaD.tensor[:2])
        print(np.sum(LambdaD_shrinked.tensor**2) / np.sum(LambdaD.tensor**2))
        GammaC_shrinked = tn.Node(GammaC.tensor[:2, :, :4])
        GammaD_shrinked = tn.Node(GammaD.tensor[:4, :, :2])
        C = bops.contract(LambdaD_shrinked, GammaC_shrinked, '1', '0', isDiag1=True)
        D = bops.contract(LambdaC_shrinked, GammaD_shrinked, '1', '0', isDiag1=True)
        C_4 = tn.Node(
            np.kron(C.tensor, np.kron(C.tensor.conj(), np.kron(C.tensor, C.tensor.conj()))))
        D_4 = tn.Node(
            np.kron(D.tensor, np.kron(D.tensor.conj(), np.kron(D.tensor, D.tensor.conj()))))
        for ti in range(len(thetas)):
            theta = thetas[ti] * np.pi / 4
            for pi in range(len(phis)):
                phi = phis[pi] * np.pi / 4
                paulis = [ft.reduce(np.matmul,
                    [linalg.expm(1j * theta * Z), linalg.expm(1j * phi * X), op, linalg.expm(-1j * phi * X), linalg.expm(-1j * theta * Z)])
                    for op in [X, Y, Z]]
                op_tensor = np.kron(I, np.kron(I, np.kron(I, I)))
                op_4 = tn.Node(op_tensor)
                for pauli in paulis: op_tensor += np.kron(pauli, np.kron(pauli, np.kron(pauli, pauli)))
                magic_C = bops.contract(bops.contract(C_4, op_4, '1', '0'), C_4, '2', '1*').tensor.transpose([0, 2, 1, 3]) \
                    .reshape([C_4[0].dimension ** 2, C_4[2].dimension ** 2])
                magic_D = bops.contract(bops.contract(D_4, op_4, '1', '0'), D_4, '2', '1*').tensor.transpose([0, 2, 1, 3]) \
                    .reshape([D_4[0].dimension ** 2, D_4[2].dimension ** 2])
                magic_CD = np.matmul(magic_C, magic_D)
                vals = np.linalg.eigvals(magic_CD)
                non_normalized_m2s[ti, pi] = -np.log(np.amax(np.abs(vals)))
        pickle.dump([GammaC, LambdaC, GammaD, LambdaD, non_normalized_m2s], open(results_filename, 'wb'))
    m2s[li] = non_normalized_m2s[0, 0]
    m2s_mins[li] = np.amin(non_normalized_m2s)
    LambdaC.tensor /= np.sqrt(np.sum(LambdaC.tensor**2))
    p2s_C[li] = np.sum(np.abs(LambdaC.tensor**4))
    p2s_D[li] = np.sum(np.abs(LambdaD.tensor**4))
    print(LambdaD.tensor)
    dbg = 1
plt.plot(ising_lambdas, m2s)
plt.plot(ising_lambdas, m2s_mins)
plt.plot(ising_lambdas, p2s_C)
plt.plot(ising_lambdas, p2s_D)
plt.show()