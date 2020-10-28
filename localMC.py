import randomUs as ru
import ising
import sys

M = int(sys.argv[1])
l = int(sys.argv[2])
chi = int(sys.argv[3])
dirname = sys.argv[4]

ru.localUnitariesMC(l, M, ising.A, ising.xRight, ising.xLeft, ising.upRow, ising.upRow,
                    dirname + 'localMC_l_' + str(l) + '_M_' + str(M), chi)
b = 1

