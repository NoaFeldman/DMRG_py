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
dirname = sys.argv[4]

with open(dirname + 'toricBoundaries', 'rb') as f:
    [upRow, downRow, leftRow, rightRow, openA, openB] = pickle.load(f)

norm = toricCode.applyLocalOperators(upRow, downRow, leftRow, rightRow, openA, openB, l,
                               [tn.Node(np.eye(d)) for i in range(l * 4)])
leftRow = bops.multNode(leftRow, 1 / norm)
ru.localUnitariesMC(l * 4, M, toricCode.applyLocalOperators, [upRow, downRow, leftRow, rightRow, openA, openB, l],
                      dirname + 'toric_local_MC', chi)

