import sys
import DMRG
import time
import numpy as np
import basicOperations as bops
import pickle

N = int(sys.argv[1])
outdir = sys.argv[2]
backend = sys.argv[3]
dev = sys.argv[4]
numOfRepetitions = sys.argv[5]

bops.init('pytorch', 'cpu')

XX = np.zeros((4, 4), dtype=complex)
XX[1, 2] = 1
XX[2, 1] = 1
Z = np.zeros((2, 2), dtype=complex)
Z[0, 0] = 1
Z[1, 1] = -1
H = DMRG.getDMRGH(N, [np.copy(Z) * 0.5 for i in range(N)], [np.copy(XX) for i in range(N - 1)])
psi0 = bops.getStartupState(N, mode='antiferromagnetic')
HLs, HRs = DMRG.getHLRs(H, psi0)

times = np.zeros(numOfRepetitions)
bondDims = np.zeros(numOfRepetitions)
for i in range(numOfRepetitions):
    start = time.time()
    gs, E0, truncErrs = DMRG.getGroundState(H, HLs, HRs, psi0)
    end = time.time()
    times[i] = end - start
    bondDims[i] = gs[int(N/2)].edges[0].dimension
with open(outdir + '/DMRG_constChi_N_' + str(N) + '_' + backend + '_' + dev, 'wb') as f:
    pickle.dump([times, bondDims], f)