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

J = -1
lambda_step = 0.1
lambda_critical_step = 0.01
phase_transition = 1
ising_lambdas = [np.round(lambda_step * i, 8) for i in range(8, int(phase_transition / lambda_step))] \
    + [np.round(phase_transition + lambda_critical_step * i, 8) for i in range(-9, 10)] \
    + [np.round(lambda_step * i, 8) for i in range(int((phase_transition + lambda_step) / lambda_step), int(1.4 / lambda_step))]
I = np.eye(2)
Z = np.diag([1, -1])
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]])

for li in range(len(ising_lambdas)):
    ising_lambda = ising_lambdas[li]
    print(ising_lambda)
    results_filename = 'magic/results/bmps_ising_lambda_' + str(ising_lambda)
    if os.path.exists(results_filename):
        [GammaC, LambdaC, GammaD, LambdaD] = pickle.load(open(results_filename, 'rb'))
    else:
        H_term = J * np.kron(Z, Z) + ising_lambda * np.kron(X, np.eye(2))
        imag_time_term = linalg.expm(H_term).reshape([1, 2, 2, 2, 2, 1]).transpose([0, 1, 3, 2, 4, 5])
        A, B, te = bops.svdTruncation(tn.Node(imag_time_term), [0, 1, 2], [3, 4, 5], '>>')
        A = tn.Node(A.tensor.transpose([1, 3, 2, 0]))
        B = tn.Node(B.tensor.transpose([1, 3, 2, 0]))
        initial_c_tensor = np.zeros((2, 2, 2), dtype=complex)
        initial_c_tensor[0, 0, 0] = 1
        initial_c_tensor[1, 1, 1] = 1
        initial_c = tn.Node(initial_c_tensor)
        initial_lambda = tn.Node(np.array([1, 1], dtype=complex))

        GammaC, LambdaC, GammaD, LambdaD = peps.getBMPSRowOps(initial_c, initial_lambda, initial_c, initial_lambda, A, B, 1000, 128)
        results_filename = 'magic/results/bmps_ising_lambda_' + str(ising_lambda)
        pickle.dump([GammaC, LambdaC, GammaD, LambdaD], open(results_filename, 'wb'))
    LambdaC.tensor = LambdaC.tensor[:4]
    LambdaD.tensor = LambdaD.tensor[:4]
    GammaC.tensor = GammaC.tensor[:4, :, :4]
    GammaD.tensor = GammaD.tensor[:4, :, :4]
    C = bops.contract(LambdaD, GammaC, '1', '0', isDiag1=True)
    D = bops.contract(LambdaC, GammaD, '1', '0', isDiag1=True)
    pickle.dump([GammaC, LambdaC, GammaD, LambdaD], open(results_filename, 'wb'))
    C_4 = tn.Node(
        np.kron(C.tensor, np.kron(C.tensor.conj(), np.kron(C.tensor, C.tensor.conj()))))
    D_4 = tn.Node(
        np.kron(D.tensor, np.kron(D.tensor.conj(), np.kron(D.tensor, D.tensor.conj()))))
    op_4 = tn.Node(np.kron(I, np.kron(I, np.kron(I, I))) + \
                   np.kron(X, np.kron(X, np.kron(X, X))) + \
                   np.kron(Y, np.kron(Y, np.kron(Y, Y))) + \
                   np.kron(Z, np.kron(Z, np.kron(Z, Z))))
    magic_C = bops.contract(bops.contract(C_4, op_4, '1', '0'), C_4, '2', '1*').tensor.transpose([0, 2, 1, 3]) \
        .reshape([C_4[0].dimension ** 2, C_4[2].dimension ** 2])
    magic_D = bops.contract(bops.contract(D_4, op_4, '1', '0'), D_4, '2', '1*').tensor.transpose([0, 2, 1, 3]) \
        .reshape([D_4[0].dimension ** 2, D_4[2].dimension ** 2])
    magic_CD = np.matmul(magic_C, magic_D)
    vals = np.lianlg.eigvals(magic_CD)
    pickle.dump([GammaC, LambdaC, GammaD, LambdaD, magic_C, magic_D, magic_CD, vals], open(results_filename, 'wb'))
    dbg = 1
