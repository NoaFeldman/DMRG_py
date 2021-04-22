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
if option == 'ising':
    magField = np.round(float(sys.argv[8]), 1)
    if int(magField) == magField:
        magField = int(magField)
    boundaryFile = 'bmpsResults_' + str(magField)
    newdir = dirname + 'ising_hf_' + str(magField) + '_n_' + str(n) + '_w_' + str(w) + '_h_' + str(h)
elif option == 'toric':
    boundaryFile = 'toricBoundaries'
    newdir = dirname + 'toric_n_' + str(n) + '_w_' + str(w) + '_h_' + str(h)

with open(dirname + boundaryFile, 'rb') as f:
    [upRow, downRow, leftRow, rightRow, openA, openB, A, B] = pickle.load(f)

[cUp, dUp, te] = bops.svdTruncation(upRow, [0, 1], [2, 3], '>>')
[cDown, dDown, te] = bops.svdTruncation(downRow, [0, 1], [2, 3], '>>')

norm = pe.applyLocalOperators(cUp, dUp, cDown, dDown, leftRow, rightRow, A, B, w, h,
                               [tn.Node(np.eye(d)) for i in range(w * h)])
leftRow = bops.multNode(leftRow, 1 / np.sqrt(norm))

try:
    os.mkdir(newdir)
except FileExistsError:
    pass
option = 'complex'
ru.renyiEntropy(n, w, h, M, option, pe.applyLocalOperators, [cUp, dUp, cDown, dDown, leftRow, rightRow, A, B, w, h],
                      newdir + '/rep_' + str(rep))
# ru.renyiNegativity(n, l * 4, M, option, toricCode.applyLocalOperators, [cUp, dUp, cDown, dDown, leftRow, rightRow, toricCode.A, toricCode.B, l],
#                    newdir + '/')