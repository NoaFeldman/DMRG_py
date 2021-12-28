import numpy as np
import tensornetwork as tn
import pickle
import jax.numpy as jnp
import torch
import sys
# sys.path.insert(1, '/home/noa/PycharmProjects/DMRG_py')
import basicOperations as bops
import time
import os
import ctypes
import gc
from  scipy import io

chi = int(sys.argv[1])
d = int(sys.argv[2])
outdir = sys.argv[3]
opt = sys.argv[4]
repetitionNum = int(sys.argv[5])
cpuNum = int(sys.argv[6])
backend = sys.argv[7]

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


def torchRandomTensor(chi, d):
    return torch.rand((chi, d, chi), dtype=torch.complex128)


def init(backend):
    if backend == 'numpy' or backend == 'numpy_nomkl':
        bops.init('numpy')
        # return tn.Node(torchTensor1.numpy()), tn.Node(torchTensor2.numpy())
    elif backend == 'jax':
        bops.init(backend)
        # return tn.Node(jnp.array(torchTensor1.numpy())), tn.Node(jnp.array(torchTensor2.numpy()))
    elif backend == 'torch_cpu':
        bops.init('pytorch', 'cpu')
    elif backend == 'torch_cuda':
        bops.init('pytorch', 'cuda')
    # elif backend == 'numpy_bare' or backend == 'numpy_bare_nomkl':
    #     return torchTensor1.numpy(), torchTensor2.numpy()
    # return tn.Node(torchTensor1), tn.Node(torchTensor2)

def singleAttempt():
    curr1Tensor = prng.random_sample((chi, d, chi)) + 1j * prng.random_sample((chi, d, chi))
    curr2Tensor = prng.random_sample((chi, d, chi)) + 1j * prng.random_sample((chi, d, chi))
    if backend == 'jax':
        curr1Tensor = jnp.array(curr1Tensor)
        curr2Tensor = jnp.array(curr2Tensor)
    if backend != 'numpy_bare':
        curr1 = tn.Node(curr1Tensor)
        curr2 = tn.Node(curr2Tensor)
    else:
        curr1 = curr1Tensor
        curr2 = curr2Tensor
    start = time.time()
    if opt == 'contraction':
        if backend == 'numpy_bare':
            c = np.tensordot(curr1, np.conj(curr2), axes=([0, 1], [0, 1]))
        else:
            c = bops.multiContraction(curr1, curr2, '01', '01')
    elif opt == 'svd':
        if backend == 'numpy_bare':
            u, s, vh = np.linalg.svd(curr1.reshape([curr1.shape[0] * curr1.shape[1], curr1.shape[2]]), full_matrices=False)
            u.reshape(curr1.shape[0], curr1.shape[1], u.shape[1])
        else:
            u, s, v, vals = tn.split_node_full_svd(curr1, curr1[:2], curr1[2:])
    end = time.time()
    return end - start

set_mkl_threads(cpuNum)
init(backend)
t = 0
prng = np.random.RandomState()
with open('random_state', 'rb') as f:
    state = pickle.load(f)
prng.set_state(state)
for rep in range(repetitionNum):
    t += singleAttempt()
    gc.collect()
filename = outdir + '/' + backend + '/' + opt + '/' + \
          'chi_' + str(chi) + '_d_' + str(d) + '_cpuNum_' + str(cpuNum) + '_repsNum_' + str(repetitionNum)
with open(filename, 'wb') as f:
    pickle.dump(t, f)
