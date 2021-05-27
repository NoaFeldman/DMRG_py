import randomUs as ru
import sys
import pickle
import tensornetwork as tn
import numpy as np
import basicOperations as bops
import os
import pepsExpect as pe


d = 2
M = int(sys.argv[1])
w = int(sys.argv[2])
h = int(sys.argv[3])
n = int(sys.argv[4])
rep = int(sys.argv[5])
dirname = sys.argv[6]
option = sys.argv[7]
theta = None
phi = None
if option == 'ising':
    magField = np.round(float(sys.argv[8]), 1)
    if int(magField) == magField:
        magField = int(magField)
    boundaryFile = 'bmpsResults_' + str(magField)
    theta = float(sys.argv[9]) * np.pi
    phi = float(sys.argv[10]) * np.pi
    newdir = dirname + 'ising_hf_' + str(magField) + '_n_' + str(n) + '_w_' + str(w) + '_h_' + str(h) \
             + '_theta_' + sys.argv[9] +'_phi_' + sys.argv[10]
    argvl = 11
elif option == 'toric':
    boundaryFile = 'toricBoundaries'
    theta = float(sys.argv[8]) * np.pi
    phi = float(sys.argv[9]) * np.pi
    newdir = dirname + 'toric_n_' + str(n) + '_w_' + str(w) + '_h_' + str(h) + '_theta_' + sys.argv[8] +'_phi_' + sys.argv[9]
    argvl = 10
elif option == 'toricG':
    g = np.round(float(sys.argv[8]), 2)
    boundaryFile = 'toricBoundaries_g_' + str(g)
    topoOpt = sys.argv[9]
    newdir = dirname + 'toric_g_' + str(g) + '_n_' + str(n) + '_w_' + str(w) + '_h_' + str(h) + '_' + topoOpt
    argvl = 10

excludeIndices = []
if len(sys.argv) > argvl:
    for i in range(argvl, len(sys.argv)):
        excludeIndices.append(int(sys.argv[i]))

with open(dirname + boundaryFile, 'rb') as f:
    [upRow, downRow, leftRow, rightRow, openA, openB, A, B] = pickle.load(f)

[cUp, dUp, te] = bops.svdTruncation(upRow, [0, 1], [2, 3], '>>')
[cDown, dDown, te] = bops.svdTruncation(downRow, [0, 1], [2, 3], '>>')

norm = pe.applyLocalOperators(cUp, dUp, cDown, dDown, leftRow, rightRow, A, B, w, h,
                               [tn.Node(np.eye(d)) for i in range(w * h)])
leftRow = bops.multNode(leftRow, 1 / norm**(2/w))
print(pe.applyLocalOperators(cUp, dUp, cDown, dDown, leftRow, rightRow, A, B, w, h,
                               [tn.Node(np.eye(d)) for i in range(w * h)]))

try:
    os.mkdir(newdir)
except FileExistsError:
    pass
option = 'complex'
ru.renyiEntropy(n, w, h, M, option, theta, phi, pe.applyLocalOperators, [cUp, dUp, cDown, dDown, leftRow, rightRow, A, B, w, h],
                      newdir + '/rep_' + str(rep), excludeIndices=excludeIndices)
# ru.renyiNegativity(n, l * 4, M, option, toricCode.applyLocalOperators, [cUp, dUp, cDown, dDown, leftRow, rightRow, toricCode.A, toricCode.B, l],
#                    newdir + '/')