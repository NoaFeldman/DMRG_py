import randomUs as ru
import sys
import pickle
import toricCode
import tensornetwork as tn
import numpy as np
import basicOperations as bops
import os


d = 2
M = int(sys.argv[1])
l = int(sys.argv[2])
option = sys.argv[3]
n = int(sys.argv[4])
if len(sys.argv) == 6:
    dirname = sys.argv[5]
else:
    dirname = 'results/'

with open(dirname + 'toricBoundaries', 'rb') as f:
    [upRow, downRow, leftRow, rightRow, openA, openB] = pickle.load(f)

[cUp, dUp, te] = bops.svdTruncation(upRow, [0, 1], [2, 3], '>>')
[cDown, dDown, te] = bops.svdTruncation(downRow, [0, 1], [2, 3], '>>')

norm = toricCode.applyLocalOperators(cUp, dUp, cDown, dDown, leftRow, rightRow, toricCode.A, toricCode.B, l,
                               [tn.Node(np.eye(d)) for i in range(l * 4)])
leftRow = bops.multNode(leftRow, 1 / norm)

newdir = dirname + option + str(n) + str(l)
try:
    os.mkdir(newdir)
except FileExistsError:
    pass
ru.renyiEntropy(n, l * 4, M, option, toricCode.applyLocalOperators, [cUp, dUp, cDown, dDown, leftRow, rightRow, toricCode.A, toricCode.B, l],
                      newdir + '/toric_local_vecs')
# ru.renyiNegativity(n, l * 4, M, option, toricCode.applyLocalOperators, [cUp, dUp, cDown, dDown, leftRow, rightRow, toricCode.A, toricCode.B, l],
#                    newdir + '/')