import numpy as np
import tensornetwork as tn
import basicDefs
import basicOperations as bops
import pepsExpect as pe
import PEPS as peps
import pickle
import randomUs as ru
from typing import List
import os
import sys

def get_boundaries(g=0.0):
    d = 2
    sites_per_node = 4
    tensor_base_site = np.array([1] + [0] * (d**sites_per_node - 1)).reshape([d] * sites_per_node)
    base_site = tn.Node(tensor_base_site)

    Xs = np.eye(1)
    for si in range(sites_per_node):
        Xs = np.kron(Xs, basicDefs.pauli2X)
    tensor_full_op = (np.eye(d**sites_per_node, dtype=complex) + Xs).reshape([d] * 2 * sites_per_node)
    full_op = tn.Node(tensor_full_op)
    tensor_splitted_op = np.zeros((d, d, 2), dtype=complex)
    tensor_splitted_op[:, :, 0] = np.eye(d)
    tensor_splitted_op[:, :, 1] = basicDefs.pauli2X
    splitted_op = tn.Node(tensor_splitted_op)

    A = tn.Node(bops.permute(bops.contract(bops.contract(bops.contract(bops.contract(bops.contract(
        base_site, full_op, '0123', '0123'),
        splitted_op, '0', '0'), splitted_op, '0', '0'), splitted_op, '0', '0'), splitted_op, '0', '0'),
        [1, 3, 5, 7, 0, 4, 2, 6]).tensor.reshape([2, 2, 2, 2, d**sites_per_node]))
    B = tn.Node(bops.contract(bops.contract(bops.contract(bops.contract(bops.contract(
        base_site, splitted_op, '0', '0'), splitted_op, '0', '0'), splitted_op, '0', '0'), splitted_op, '0', '0'),
        full_op, '0246', '0123').tensor.reshape([2, 2, 2, 2, d**sites_per_node]))


    single_site_tension = tn.Node(np.diag([(1+g)**(bin(i).count('1')) for i in range(d**sites_per_node)]))
    A = bops.contract(A, single_site_tension, '4', '0')
    B = bops.contract(B, single_site_tension, '4', '0')


    AEnv = pe.toEnvOperator(bops.multiContraction(A, A, '4', '4*'))
    BEnv = pe.toEnvOperator(bops.multiContraction(B, B, '4', '4*'))
    upRow, downRow, leftRow, rightRow, openA, openB = peps.applyBMPS(A, B, d=d**sites_per_node)
    with open('results/toricBoundaries_squares_' + str(g), 'wb') as f:
        pickle.dump([upRow, downRow, leftRow, rightRow, openA, openB, A, B], f)
    circle_A = bops.contract(bops.contract(bops.contract(bops.contract(bops.contract(bops.contract(bops.contract(
        leftRow, upRow, '3', '0'), BEnv, '23', '30'), AEnv, '41', '30'), rightRow, '24', '01'), BEnv, '12', '03'),
        downRow, '053', '320'), openA, '0132', '1234')
    b = 1


def get_random_ops(n, N, theta=0, phi=0, random_option='full'):
    rotating_op = np.matmul(ru.getUPhi(phi), ru.getUTheta(theta))
    single_site_ops = ru.getNonUnitaryRandomOps(2, n, N * 2, random_option=random_option)
    return [[tn.Node(np.kron(np.matmul(np.matmul(rotating_op, single_site_ops[ni][Ni * 2].tensor), rotating_op.T.conj()),
                             np.kron(np.eye(2),
                    np.kron(np.matmul(np.matmul(rotating_op, single_site_ops[ni][Ni * 2 + 1].tensor), rotating_op.T.conj()),
                            np.eye(2)))))
             for Ni in range(N)] for ni in range(n)]


w = int(sys.argv[1])
h = int(sys.argv[2])
n = int(sys.argv[3])
rep = sys.argv[4]
indir = sys.argv[5]
if len(sys.argv) > 6:
    exclude_indicies = [int(arg) for arg in sys.argv[6:]]
else:
    exclude_indicies = []
N = w * h
with open(indir + '/toricBoundaries_squares_' + str(0.0), 'rb') as f:
    [upRow, downRow, leftRow, rightRow, openA, openB, A, B] = pickle.load(f)
[cUp, dUp, te] = bops.svdTruncation(upRow, [0, 1], [2, 3], '>>')
[cDown, dDown, te] = bops.svdTruncation(downRow, [0, 1], [2, 3], '>>')
norm = pe.applyLocalOperators(cUp, dUp, cDown, dDown, leftRow, rightRow, A, B, w, h,
                              [tn.Node(np.eye(16)) for i in range(w * h)])
leftRow = bops.multNode(leftRow, 1 / norm ** (2 / w))

newdir = indir + '/toric_checkerboard/w_' + str(w) + '_h_' + str(h) + '_n_' + str(n) \
         + '_excluded_' + str(exclude_indicies)
try:
    os.mkdir(newdir)
except FileExistsError:
    pass

M = 1000
ru.renyiEntropy(n, w, h, M, pe.applyLocalOperators, [cUp, dUp, cDown, dDown, leftRow, rightRow, A, B, w, h],
                      newdir + '/rep_' + str(rep), d=2**4, excludeIndices=exclude_indicies,
                get_ops_func=get_random_ops, get_ops_arguments=[n, N, 0, 0, 'full'])
