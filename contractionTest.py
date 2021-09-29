import numpy as np
import tensornetwork as tn
import pickle
import jax
import jax.numpy as jnp
import torch
import sys
# sys.path.insert(1, '/home/noa/PycharmProjects/DMRG_py')
import basicOperations as bops
import time


chi = int(sys.argv[1])
d = int(sys.argv[2])
outdir = sys.argv[3]
opt = sys.argv[4]
repetitions = int(sys.argv[5])


def torchRandomTensor(chi, d):
    return torch.rand((chi, d, chi), dtype=torch.complex128)


def prepare(torchTensor1, torchTensor2, backend):
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
    return tn.Node(torchTensor1), tn.Node(torchTensor2)


tSums = {'numpy' : 0, 'jax': 0, 'torch_cpu': 0, 'torch_cuda': 0}
backends = list(tSums.keys())
n = len(backends)
for rep in range(repetitions):
    currTorch1 = torchRandomTensor(chi, d)
    currTorch2 = torchRandomTensor(chi, d)
    for backend in [backends[(rep + i) % n] for i in range(n)]:
        curr1, curr2 = prepare(currTorch1, currTorch2, backend)
        start = time.time()
        if opt == 'contraction':
            bops.multiContraction(curr1, curr2, '01', '01')
            print(bops.multiContraction(curr1, curr2, '01', '01').tensor)
        elif opt == 'svd':
            tn.split_node_full_svd(curr1, curr1.edges[:2], curr1.edges[2:])
            print(tn.split_node_full_svd(curr1, curr1.edges[:2], curr1.edges[2:])[1].tensor)
        end = time.time()
        tSums[backend] += end - start
for backend in backends:
    with open(outdir + '/' + backend + '_' + opt + '_' + str(chi) + '_' + str(d) + '_' + str(repetitions), 'wb') as f:
        pickle.dump(tSums[backend], f)