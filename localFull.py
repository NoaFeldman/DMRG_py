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
if len(sys.argv) == 4:
    dirname = sys.argv[3]
else:
    dirname = ''

with open(dirname + 'toricBoundaries', 'rb') as f:
    [upRow, downRow, leftRow, rightRow, openA, openB] = pickle.load(f)

[cUp, dUp, te] = bops.svdTruncation(upRow, [0, 1], [2, 3], '>>')
[cDown, dDown, te] = bops.svdTruncation(downRow, [0, 1], [2, 3], '>>')

norm = toricCode.applyLocalOperators(cUp, dUp, cDown, dDown, leftRow, rightRow, toricCode.A, toricCode.B, l,
                               [tn.Node(np.eye(d)) for i in range(l * 4)])
leftRow = bops.multNode(leftRow, 1 / norm)
ru.localUnitariesFull(l * 4, M, toricCode.applyLocalOperators, [cUp, dUp, cDown, dDown, leftRow, rightRow, toricCode.A, toricCode.B, l],
                      dirname + 'toric_local_full')
