import numpy as np
import tensornetwork as tn
import pickle
import jax
import torch
import sys
# sys.path.insert(1, '/home/noa/PycharmProjects/DMRG_py')
import basicOperations as bops
import time


chi = int(sys.argv[1])
d = int(sys.argv[2])
outdir = sys.argv[3]
backend = sys.argv[4]
if sys.argv[5] == 'None':
    dev = None
else:
    dev = sys.argv[5]
opt = sys.argv[6]
repetitions = int(sys.argv[7])


def numpyRandomTensor(chi, d):
    return np.random.rand(chi, d, chi) + 1j * np.random.rand(chi, d, chi)


def torchRandomTensor(chi, d):
    return torch.rand((chi, d, chi), dtype=torch.complex64)


def defaultRandomTensor(chi, d):
    print('Backend unsupported!')
    return 0


bops.init(backend, dev)
if backend == 'numpy':
    getRandomTensor = numpyRandomTensor
elif backend == 'jax':
    getRandomTensor = numpyRandomTensor
elif backend == 'pytorch':
    getRandomTensor = torchRandomTensor
else:
    getRandomTensor = defaultRandomTensor


if opt == 'contraction':
    start = time.time()
    for i in range(repetitions):
        bops.multiContraction(tn.Node(getRandomTensor(chi, d)), tn.Node(getRandomTensor(chi, d)), '01', '01*')
    end = time.time()
elif opt == 'svd':
    start = time.time()
    for i in range(repetitions):
        node = tn.Node(getRandomTensor(chi, d))
        tn.split_node_full_svd(node, node.edges[:2], [node.edges[2]])
    end = time.time()
else:
    print('opt unsupported!')
    start = 0
    end = 0
with open(outdir + '/' + opt + '_chi_' + str(chi) + '_d_' + str(d) + '_reps_' + str(repetitions), 'wb') as f:
        pickle.dump(end - start, f)
