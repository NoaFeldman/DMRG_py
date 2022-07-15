import numpy as np
import pickle
import tdvp
import sys
import os
import tensornetwork as tn
import basicOperations as bops
import scipy.linalg as linalg
import swap_dmrg as swap
from typing import List


# Controlled bond expansion for DMRG ground state search at single-site costs
# Andreas Gleis,1 Jheng-Wei Li,1 and Jan von Delft

# Fig 2
def get_op_tilde_tr(k: int, HL: List[tn.Node], HR: List[tn.Node], H: List[tn.Node],
                    A: tn.Node, Delta: tn.Node, B: tn.Node, dir: str, D:int, w:int) -> tn.Node:
    a_left_id = bops.contract(bops.contract(HL[k], A, '0', '0'), H[k], '02', '20')
    a_left_Al = bops.contract(bops.contract(a_left_id, A, '02', '01*'), A, '2', '2')
    a_left = tn.Node(a_left_id.tensor.tranpose([1, 3, 0, 2]) - a_left_Al.tensor)

    a_right_id = bops.contract(bops.contract(HR[k+1], B, '0', '2'), H[k+1], '03', '30')
    a_right_Bl1 = bops.contract(bops.contract(a_right_id, B, '02', '21*'), B, '2', '0')
    a_right = tn.node(a_right_id.tensor.transpose([1, 3, 2, 0]) - a_right_Bl1.tensor)

    if dir == '>>':
        pink = bops.contract(Delta, a_right, '1', '0')
        [US, V, te] = bops.svdTruncation(pink, [0], [1, 2, 3], '<<', maxBondDim=D)
        blue = bops.contract(a_left, US, '0', '0')
        [red, V, te] = bops.svdTruncation(blue, [0, 1, 2], [3], '<<', maxBondDim=int(D/w))
        [red_site, S, V, te] = bops.svdTruncation(red, [1, 2], [0, 3], '>*<')
        yellow = bops.contract(red_site, bops.contract(pink, a_left_id, '01', '13'), '01*', '23')
        [u_tilde, s_tilde, v_tilde, te] = bops.svdTruncation(yellow, [0], [1, 2], '>*<')
        yellow_site = bops.contract(red_site, u_tilde, '2', '0')
        green_site_tensor = np.zeros((A.tensor.shape[0], A.tensor.shape[1], A.tensor.shape[2] + yellow_site.tensor.shape[2]), dtype=complex)
        green_site_tensor[:, :, :A.tensor.shape[2]] = A.tensor
        green_site_tensor[:, :, A.tensor.shape[2]:] = yellow_site.tensor
        green_site = tn.Node(green_site_tensor)
        green_C = bops.contract(bops.contract(green_site, A, '01*', '01'), Delta, '1', '0')
    else: # dir == '<<':
        pink = bops.contract(a_left, Delta, '0', '0')
        [U, SV, te] = bops.svdTruncation(pink, [0, 1, 2], [3], '>>', maxBondDim=D)
        blue = bops.contract(SV, a_right, '1', '0')
        [U, red, te] = bops.svdTruncation(blue, [0], [1, 2, 3], '>>', maxBondDim=int(D/w))
        [U, S, red_site] = bops.svdTruncation(red, [0, 1], [2, 3], '>*<')
        yellow = bops.contract(bops.contract(pink, a_right_id, '30', '13'), red_site, '32', '12*')
        [u_tilde, s_tilde, v_tilde] = bops.svdTruncation(yellow, [0, 1], [2], '>*<')
        yellow_site = bops.contract(v_tilde, red_site, '1', '0')
        green_site = np.zeros((B.tensor.shape[0] + yellow_site.tensor.shape[0], B.tensor.shape[1], B.tensor.shape[2]), dtype=complex)
        green_site[:, :, :A.tensor.shape[2]] = A.tensor
        green_site[:, :, A.tensor.shape[2]:] = yellow_site.tensor
    return tn.Node(result)
