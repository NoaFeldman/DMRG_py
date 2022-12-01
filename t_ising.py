import basicOperations as bops
import numpy as np
import DMRG as dmrg
import pickle
import basicDefs
import sys


N = int(sys.argv[1])
J = int(sys.argv[2])
ising_lambda = np.round(float(sys.argv[3]), 3)
dirname = sys.argv[4]

onsite_term = ising_lambda * basicDefs.pauli2X
neighbor_term = J * np.kron(basicDefs.pauli2Z, basicDefs.pauli2Z)

psi_0 = bops.getStartupState(N, 2)
gs, E0, trunc_errs = dmrg.DMRG(psi_0, [onsite_term] * N, [neighbor_term] * (N - 1), maxBondDim=1024)
pickle.dump([gs, E0, trunc_errs], open(dirname + '/t_ising_gs_N_' + str(N) + '_J_' + str(J) + '_lambda_' + str(ising_lambda)))