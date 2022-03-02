import numpy as np
import tensornetwork as tn
import basicOperations as bops
import time
import pickle
import os
import randomUs as ru
import pepsExpect as pe
import DMRG as dmrg
import basicDefs as basic
import sys


def set_mkl_threads(threadNum):
    try:
        import mkl
        mkl.set_num_threads(threadNum)
        return 0
    except:
        pass

    for name in ["libmkl_rt.so", "libmkl_rt.dylib", "mkl_Rt.dll"]:
        try:
            mkl_rt = ctypes.CDLL(name)
            mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(1)))
            return 0
        except:
            pass

    os.environ["OMP_NUM_THREADS"] = str(threadNum)  # export OMP_NUM_THREADS=4
    os.environ["OPENBLAS_NUM_THREADS"] = str(threadNum)  # export OPENBLAS_NUM_THREADS=4
    os.environ["MKL_NUM_THREADS"] = str(threadNum)  # export MKL_NUM_THREADS=6
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(threadNum)  # export VECLIB_MAXIMUM_THREADS=4
    os.environ["NUMEXPR_NUM_THREADS"] = str(threadNum)  # export NUMEXPR_NUM_THREADS=6



def test_toric_code_moment_estimation(N, dirname):
    boundaryFile = 'toricBoundaries'
    with open(dirname + boundaryFile, 'rb') as f:
        [upRow, downRow, leftRow, rightRow, openA, openB, A, B] = pickle.load(f)

    [cUp, dUp, te] = bops.svdTruncation(upRow, [0, 1], [2, 3], '>>')
    [cDown, dDown, te] = bops.svdTruncation(downRow, [0, 1], [2, 3], '>>')
    newdir = dirname + '/test'
    try:
        os.mkdir(newdir)
    except FileExistsError:
        pass
    n = 2
    w = 2
    h = int(N / (2 * w))
    M = 1000
    estimate_func = pe.applyLocalOperators
    ru.renyiEntropy(n, w, h, M, estimate_func, [cUp, dUp, cDown, dDown, leftRow, rightRow, A, B, w, h],
                          newdir + '/test')


def test_dmrg_xxz(N):
    psi = bops.getStartupState(N, 2)
    delta = 0.5
    onsite_terms = [0 * np.eye(2) for i in range(N)]
    neighbor_terms = [np.kron(basic.pauli2Z, basic.pauli2Z) * delta + \
                      np.kron(basic.pauli2X, basic.pauli2X) + \
                      np.kron(basic.pauli2Y, basic.pauli2Y) for i in range(N - 1)]
    psi, E0, truncErrs = dmrg.DMRG(psi, onsite_terms, neighbor_terms, accuracy=1e-12)


cpu_num = int(sys.argv[1])
set_mkl_threads(cpu_num)

opt = sys.argv[2]
result_file_name = sys.argv[3]
start = time.time()
N = int(sys.argv[4])
if opt == 'PEPS':
    dirname = sys.argv[5]
    test_toric_code_moment_estimation(N, dirname)
elif opt == 'DMRG':
    test_dmrg_xxz(N)
end = time.time()
with open(result_file_name + '_' + opt + '_N_' + str(N), 'wb') as f:
    pickle.dump(end - start)