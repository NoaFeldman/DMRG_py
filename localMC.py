import randomUs as ru
import sys
import pickle
import toricCode
import tensornetwork as tn
import numpy as np
import basicOperations as bops


d = 2
M = int(sys.argv[1])
l = int(sys.argv[2])
chi = int(sys.argv[3])
if len(sys.argv) == 5:
    dirname = sys.argv[4]
else:
    dirname = ''
M = 1000
chi = 100
with open(dirname + 'toricBoundaries', 'rb') as f:
    [upRow, downRow, leftRow, rightRow, openA, openB] = pickle.load(f)

[cUp, dUp, te] = bops.svdTruncation(upRow, [0, 1], [2, 3], '>>')
[cDown, dDown, te] = bops.svdTruncation(downRow, [0, 1], [2, 3], '>>')

norm = toricCode.applyLocalOperators(cUp, dUp, cDown, dDown, leftRow, rightRow, toricCode.A, toricCode.B, l,
                               [tn.Node(np.eye(d)) for i in range(l * 4)])
leftRow = bops.multNode(leftRow, 1 / norm)
ru.localUnitariesMC(l * 4, M, toricCode.applyLocalOperators, [cUp, dUp, cDown, dDown, leftRow, rightRow, toricCode.A, toricCode.B, l],
                      dirname + 'toric_local_MC', chi)

