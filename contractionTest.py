import numpy as np
import tensornetwork as tn
import pickle
# import jax
# import jax.numpy as jnp
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


def prepare(torchTensor1, torchTensor2, backend, cpuNum):
    if backend == 'numpy':
        bops.init(backend)
        return tn.Node(torchTensor1.numpy()), tn.Node(torchTensor2.numpy())
    elif backend == 'jax':
        bops.init(backend)
        return tn.Node(jnp.array(torchTensor1.numpy())), tn.Node(jnp.array(torchTensor2.numpy()))
    elif backend == 'torch_cpu':
        bops.init('pytorch', 'cpu')
    elif backend == 'torch_cuda':
        bops.init('pytorch', 'cuda')
    elif backend == 'numpy_bare':
        return torchTensor1.numpy(), torchTensor2.numpy()
    return tn.Node(torchTensor1), tn.Node(torchTensor2)

def singleAttempt():
    with open('randomTensors' 'rb') as f:
        [currTorch1, currTorch2] = pickle.load(f)
    for backend in [backends[(rep + i) % n] for i in range(n)]:
        curr1, curr2 = prepare(currTorch1, currTorch2, backend, cpuNum)
        start = time.time()
        if opt == 'contraction':
            if backend == 'numpy_bare':
                c = np.tensordot(curr1, np.conj(curr2), axes=([0, 1], [0, 1]))
            else:
                c = bops.multiContraction(curr1, curr2, '01', '01')
        elif opt == 'svd':
            if backend == 'numpy_bare':
                u, s, vh = np.linalg.svd(curr1.reshape([curr1.shape[0] * curr1.shape[1], curr1.shape[2]]))
                u.reshape(curr1.shape[0], curr1.shape[1], u.shape[1])
            else:
                u, s, v, vals = tn.split_node_full_svd(curr1, curr1[:2], curr1[2:])
        end = time.time()
        ts[backend] += end - start
    currTorch1 = torchRandomTensor(chi, d)
    currTorch2 = torchRandomTensor(chi, d)
    with open('randomTensors' 'wb') as f:
        pickle.dump([currTorch1, currTorch2], f)
    io.savemat('randomTensors.mat', dict(curr1=currTorch1.numpy(), curr2=currTorch2.numpy()))

set_mkl_threads(cpuNum)
ts = {'numpy': 0, 'torch_cpu': 0, 'jax': 0, 'torch_cuda': 0, 'numpy_bare': 0}
backends = list(ts.keys())
n = len(backends)
# for rep in range(repetitions):
singleAttempt()
gc.collect()
for backend in backends:
    filename = outdir + '/' + backend + '/' + opt + '/' + \
              'chi_' + str(chi) + '_d_' + str(d) + '_cpuNum_' + str(cpuNum)
    if repetitionNum == 0:
        with open(filename + '_rep_' + str(repetitionNum), 'wb') as f:
            pickle.dump(ts[backend], f)
    else:
        with open(filename + '_rep_' + str(repetitionNum - 1), 'wb') as f:
            t = pickle.load(f) + ts[backend]
        with open(filename + '_rep_' + str(repetitionNum), 'wb') as f:
            pickle.dump(t, f)
        os.remove(filename + '_rep_' + str(repetitionNum - 1))
