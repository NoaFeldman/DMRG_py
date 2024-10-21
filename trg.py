import numpy as np
import basicOperations as bops
import tensornetwork as tn
from typing import List

# https://tensornetwork.org/trg/

def parity(i, j):
    return int(i%2 != j%2)


def trg_step(network: List[List[tn.Node]], chi=8):
    x = len(network[0])
    y = len(network)
    if x == 8:
        dbg = 1
    diag_fs = [[None for i in range(x)] for j in range(y)]
    max_te = 0
    for ri in range(y):
        for ci in range(x):
            if parity(ri, ci) == 0:
                [u, s, v, te] = bops.svdTruncation(network[ri][ci], [0, 3], [1, 2], '>*<', maxBondDim=chi)
            else:
                [u, s, v, te] = bops.svdTruncation(network[ri][ci], [0, 1], [2, 3], '>*<', maxBondDim=chi)
            diag_fs[ri][ci] = [bops.contract(u, tn.Node(np.sqrt(s.tensor)), '2', '0'), bops.contract(tn.Node(np.sqrt(s.tensor)), v, '1', '0')]
            te = np.sum(te) / (np.sum(np.diag(s.tensor)) + np.sum(te))
            if max_te < te: max_te = te
    parallel_fs = [[None for i in range(x + 1)] for j in range(y + 1)]
    new_network = [[None for i in range(y + 1)] for j in range(x + 1)]
    for ri in range(y + 1):
        for ci in range(x + 1):
            if parity(ri, ci) == 0:
                f_inds = [[ri - 1, ci - 1], [ri - 1, ci], [ri, ci], [ri, ci - 1]]
                f_ops = [diag_fs[ind[0]][ind[1]] if (ind[0] >= 0 and ind[0] < y and ind[1] >= 0 and ind[1] < x) \
                             else [tn.Node(np.ones((1, 1, 1))), tn.Node(np.ones((1, 1, 1)))] for ind in f_inds]
                if ri > 3 and ci > 3:
                    dbg = 1
                new_node = bops.contract(bops.contract(bops.contract(
                    f_ops[0][1], f_ops[1][1], '1', '2'), f_ops[2][0], '3', '0'), f_ops[3][0], '13', '01')
                if ri % 2 == 0:
                    [u, s, v, te] = bops.svdTruncation(new_node, [0, 3], [1, 2], '>*<', maxBondDim=chi)
                else:
                    [u, s, v, te] = bops.svdTruncation(new_node, [0, 1], [2, 3], '>*<', maxBondDim=chi)
                new_network[ri][ci] = new_node
            parallel_fs[ri][ci] = [bops.contract(u, tn.Node(np.sqrt(s.tensor)), '2', '0'),
                                   bops.contract(tn.Node(np.sqrt(s.tensor)), v, '1', '0')]
            te = np.sum(te) / (np.sum(np.diag(s.tensor)) + np.sum(te))
            if max_te < te: max_te = te
    new_network = [[None for i in range(int(y / 2) + 2)] for j in range(int(x / 2) + 2)]
    for ri in range(int(y/2) + 2):
        for ci in range(int(x/2) + 2):
            f_inds = [[2 * ri - 1, 2 * ci - 1], [2 * ri, 2 * ci], [2 * ri + 1, 2 * ci - 1], [2 * ri, 2 * ci - 2]]
            f_ops = [parallel_fs[ind[0]][ind[1]] if (ind[0] >= 0 and ind[0] < y+1 and ind[1] >= 0 and ind[1] < x+1) \
                     else [tn.Node(np.ones((1, 1, 1))), tn.Node(np.ones((1, 1, 1)))] for ind in f_inds]
            new_node = bops.contract(bops.contract(bops.contract(
                f_ops[0][1], f_ops[1][0], '1', '0'), f_ops[2][0], '2', '1'), f_ops[3][1], '13', '12')
            new_network[ri][ci] = new_node
    return new_network, max_te


def trg(network: List[List[tn.Node]], chi=8):
    x = len(network)
    y = len(network[0])
    max_te = 0
    while x > 4 and y > 4:
        network, te = trg_step(network=network, chi=chi)
        if te > max_te: max_te = te
        x = len(network)
        y = len(network[0])
    return network, max_te