import numpy as np
import basicOperations as bops
import tensornetwork as tn
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
from typing import List
import pickle
import sys
import os
import PEPS as peps
import pepsExpect as pe
import sys

def full_inds(N, i, bc='P'):
    inds = [int(c) for c in bin(i).split('b')[1].zfill(N**2 + 1)]
    inds.reverse()
    if bc == 'P':
        inds_2d = [inds[:N]]
        for ri in range(1, N):
            inds_2d += [inds[N*ri:N*(ri+1)], [-1]*N]
        inds_2d.append([inds[-1]] + [-1] * (N-1))
        for ci in range(N):
            for ri in range(2*N - 1):
                if inds_2d[ri][ci] == -1:
                    inds_2d[ri][ci] = (inds_2d[(ri-2)%(2*N)][ci] + inds_2d[(ri-1)%(2*N)][ci] + inds_2d[(ri-1)%(2*N)][(ci-1)%N]) % 2
        for ci in range(1, N):
            inds_2d[2*N - 1][ci] = (inds_2d[2*N-2][ci] + inds_2d[2*N-1][ci - 1] + inds_2d[0][ci]) % 2
        return np.array(inds_2d).reshape([2 * N ** 2])
    else:
        inds_2d = [inds[:N] + [0], [inds[0]] + [(inds[i] + inds[i+1]) % 2 for i in range(N-1)] + [inds[N-1]]]
        for ri in range(1, N):
            inds_2d += [inds[N*ri:N*(ri+1)] + [0]]
            new_row = ([(inds_2d[-1][0] + inds_2d[-2][0]) % 2]
                       + [(inds_2d[-1][i] + inds_2d[-1][i+1] + inds_2d[-2][i+1]) % 2 for i in range(N-1)]
                       + [(inds_2d[-1][-2] + inds_2d[-2][-1])])
            inds_2d += [new_row]
        new_row = [inds_2d[-1][0]]
        for ci in range(1, N):
            new_row.append((new_row[-1] + inds_2d[-1][ci]) % 2)
        inds_2d += [new_row + [0]]
    return np.array(inds_2d).reshape([(2 * N + 1) * (N+1)])


# N*N lattice site system (2*N*N edges)
# Jp = coefficient of \prod_{e \in p} X_e
# hz = coefficient of \sum_e Z_e
# Only creating gauge-invariant block
def exact_H_Z2(N, Jp, hz, small_loop_c=0, bc='P'):
    if bc == 'P':
        block_size = 2**(N**2 + 1)
    else: # bc == 'O'
        block_size = 2**(N**2)
    H = sparse.lil_matrix((block_size, block_size))
    for col in range(N):
        for row in range(N):
            if bc == 'P':
                plaquette_free_bit_inds = []
                if row != N-1 or col == 0:
                    plaquette_free_bit_inds.append(row * N + col + N)
                if row != N-2 or col == 0:
                    plaquette_free_bit_inds.append(((row + 1) % N) * N + col + N)
                if row == N-1:
                    plaquette_free_bit_inds += [col, (col + 1) % N]
            else: # bc == 'O'
                plaquette_free_bit_inds = [row * N + col, (row + 1) * N + col] if row < N - 1 else [row * N + col]
            for i in range(block_size):
                j = i ^ sum([2**ind for ind in plaquette_free_bit_inds])
                H[i, j] = -Jp
                H[j, i] = -Jp
    if small_loop_c != 0:
        for i in range(block_size):
            f_inds = full_inds(N, i, bc)
            H[i, i] = -hz * (2*N**2 - 2 * np.sum(f_inds)) # + lambdas[li] * 2 * N**2 + N**2 / lambdas[li]
            f_inds = f_inds.reshape([2*N, N])
            for pri in range(N):
                for pci in range(N):
                    if f_inds[2 * pri][pci] == 1 and f_inds[(2 * pri + 1) % (2*N)][pci] == 1 and f_inds[2 * pri][(pci + 1) % N] == 1 and f_inds[(2 * pri - 1) % (2*N)][pci] == 1:
                        H[i, i] -= small_loop_c
    return H


# TODO this is only checked for N=2
def get_expectation_value(N: int, nodes: List[List[tn.Node]], bc='P', edge=None):
    if bc == 'P':
        column = nodes[0][0]
        for ri in range(1, N-1):
            column = bops.contract(column, nodes[ri][0], [2*ri], '0')
        column = bops.permute(bops.contract(column, nodes[-1][0], [0, 2 * (N-1)], '20'),
                              [2 * i + 1 for i in range(N)] + [2 * i for i in range(N)])
        for ci in range(1, N-1):
            column = bops.contract(column, nodes[0][ci], [N], '3')
            import gc
            gc.collect()
            for ri in range(1, N - 1):
                column = bops.contract(column, nodes[ri][ci], [N, 2 * N + 1], '30')
            column = bops.contract(column, nodes[-1][ci], [N, 2*N+1, N+1], '302')
        column = bops.contract(column, nodes[0][-1], [0, N], '13')
        for ri in range(1, N-1):
            column = bops.contract(column, nodes[ri][-1], [0, N - ri, (N - ri) * 2 + 1], '130')
        result = bops.contract(column, nodes[-1][-1], '0123', '1320')
        return result.tensor * 1
    else:
        column_tensor = edge.tensor
        for ri in range(1, N+1):
            column_tensor = np.kron(column_tensor, edge.tensor)
        column = tn.Node(column_tensor.reshape([edge[0].dimension] * (N+1)))
        for ci in range(N+1):
            column = bops.contract(column, bops.contract(nodes[0][ci], edge, '0', '0'), '0', '2')
            for ri in range(1, N):
                column = bops.contract(column, nodes[ri][ci], [0, N + 1], '30')
            column = bops.contract(column, bops.contract(nodes[-1][ci], edge, '2', '0'), [0, N+1], '20')
        for ci in range(N+1):
            column = bops.contract(column, edge, '0', '0')
        return column.tensor * 1


def minimal_ansatz_node(params: List[complex], is_sr=True):
    d = 2
    X = np.array([[0, 1], [1, 0]])
    p1, l1, p2, l2, l4 = params
    node = tn.Node(np.array([1] + [0] * (d**4 - 1)).reshape([d, d, d, d]))
    plaquette_op_tensor = np.zeros((d, d, d, d, 2), dtype=complex)
    plaquette_op_tensor[:, :, :, :, 0] = np.eye(d ** 2).reshape([d] * 4)
    plaquette_op_tensor[:, :, :, :, 1] = np.kron(X, X).reshape([d] * 4)
    plaquette = tn.Node(plaquette_op_tensor)
    node = bops.permute(bops.contract(bops.contract(bops.contract(bops.contract(
        node, plaquette, '01', '01'), plaquette, '30', '01'), plaquette, '40', '01'), plaquette, '50', '01'),
        [0, 2, 4, 7, 6, 1, 3, 5])
    node.tensor[1, :] *= np.exp(-p1)
    node.tensor[:, 1, :] *= np.exp(-p1)
    node.tensor[:, :, 1, :] *= np.exp(-p1)
    node.tensor[:, :, :, 1, :] *= np.exp(-p1)
    node.tensor[0, 1, :] *= np.exp(-l1)
    node.tensor[1, 0, :] *= np.exp(-l1)
    node.tensor[:, 0, 1, :] *= np.exp(-l1)
    node.tensor[:, 1, 0, :] *= np.exp(-l1)
    node.tensor[:, :, 0, 1, :] *= np.exp(-l1)
    node.tensor[:, :, 1, 0, :] *= np.exp(-l1)
    node.tensor[0, :, :, 1, :] *= np.exp(-l1)
    node.tensor[1, :, :, 0, :] *= np.exp(-l1)
    if is_sr:
        node.tensor[1, 1, :] *= np.exp(-p2)
        node.tensor[:, 1, 1, :] *= np.exp(-p2)
        node.tensor[:, :, 1, 1, :] *= np.exp(-p2)
        node.tensor[1, :, :, 1, :] *= np.exp(-p2)
        for i in range(2):
            for j in range(2):
                node.tensor[i, (i + 1) % 2, j, (j + 1) % 2, :] *= np.exp(-l2)
                node.tensor[j, i, (i + 1) % 2, (j + 1) % 2, :] *= np.exp(-l2)
        node.tensor[1, 0, 1, 0, :] *= np.exp(-l4)
        node.tensor[0, 1, 0, 1, :] *= np.exp(-l4)
    node = tn.Node(node.tensor.reshape([2] * 4 + [d**4]))
    return node


#   X  |  Y
#  ____|____
#   Z  |  W
#      |
# TODO Not adjusted for the new full star nodes
def get_energy(N: int, params, Jp: float, hz: float, is_sr=True, opt='full_star', bc='P'):
    node = minimal_ansatz_node(params=params, is_sr=is_sr, opt=opt)
    X = tn.Node(np.array([[0, 1], [1, 0]]))
    Z = tn.Node(np.diag([1, -1]))
    proj_1 = tn.Node(np.diag([0, 1]))
    if opt == 'lattice_site':
        area_coeff, perimeter_coeff, sr_coeff = params
        traced_node = tn.Node(bops.contract(node, node, '45', '45*').tensor.transpose([0, 4, 1, 5, 2, 6, 3, 7])
                              .reshape([4**2] * 4))
        valid_inds = [0, 3, 5, 6, 9, 10, 12, 15]
        double_leg_projector = tn.Node(np.array([[int(i == valid_inds[j]) for j in range(len(valid_inds))] for i in range(16)]))
        traced_node_full_projected = bops.contract(bops.contract(bops.contract(bops.contract(
            traced_node, double_leg_projector, '0', '0'), double_leg_projector, '0', '0'), double_leg_projector, '0', '0'), double_leg_projector, '0', '0')
        traced_node_outgoing_projected = bops.permute(bops.contract(bops.contract(
            traced_node, double_leg_projector, '0', '0'), double_leg_projector, '0', '0'), [2, 3, 0, 1])
        traced_node_xx = tn.Node(bops.contract(bops.permute(bops.contract(bops.contract(node, X, '5', '0'), X, '4', '0'),
                                                            [0, 1, 2, 3, 5, 4]),
                                               node, '45', '45*').tensor.transpose([0, 4, 1, 5, 2, 6, 3, 7])
                                 .reshape([4**2] * 4))
        traced_node_xx = bops.contract(bops.contract(traced_node_xx,
                                double_leg_projector, '2', '0'), double_leg_projector, '2', '0')
        traced_node_x_top = tn.Node(bops.contract(bops.permute(bops.contract(node, X, '4', '0'), [0, 1, 2, 3, 5, 4]),
                                               node, '45', '45*').tensor.transpose([0, 4, 1, 5, 2, 6, 3, 7])
                                 .reshape([4**2] * 4))
        traced_node_x_top = bops.permute(bops.contract(bops.contract(traced_node_x_top,
                            double_leg_projector, '1', '0'), double_leg_projector, '1', '0'), [0, 2, 3, 1])
        traced_node_x_right = tn.Node(bops.contract(bops.contract(node, X, '5', '0'),
                                               node, '45', '45*').tensor.transpose([0, 4, 1, 5, 2, 6, 3, 7])
                                 .reshape([4**2] * 4))
        traced_node_x_right = bops.permute(bops.contract(bops.contract(traced_node_x_right,
                            double_leg_projector, '0', '0'), double_leg_projector, '2', '0'), [2, 0, 1, 3])
        traced_node_z_top = tn.Node(bops.contract(bops.permute(bops.contract(node, Z, '4', '0'), [0, 1, 2, 3, 5, 4]),
                                               node, '45', '45*').tensor.transpose([0, 4, 1, 5, 2, 6, 3, 7])
                                 .reshape([4**2] * 4))
        traced_node_z_top = bops.contract(bops.contract(bops.contract(bops.contract(
            traced_node_z_top, double_leg_projector, '0', '0'), double_leg_projector, '0', '0'), double_leg_projector, '0', '0'), double_leg_projector, '0', '0')
        traced_node_z_right = tn.Node(bops.contract(bops.contract(node, Z, '5', '0'),
                                               node, '45', '45*').tensor.transpose([0, 4, 1, 5, 2, 6, 3, 7])
                                 .reshape([4**2] * 4))
        traced_node_z_right = bops.contract(bops.contract(bops.contract(bops.contract(traced_node_z_right, double_leg_projector, '0', '0'),
                                            double_leg_projector, '0', '0'), double_leg_projector, '0', '0'), double_leg_projector, '0', '0')
        traced_node_proj_top = tn.Node(bops.contract(bops.permute(bops.contract(node, proj_1, '4', '0'), [0, 1, 2, 3, 5, 4]),
                                node, '45', '45*').tensor.transpose([0, 4, 1, 5, 2, 6, 3, 7]).reshape([4**2] * 4))
        traced_node_proj_top = bops.contract(bops.contract(bops.contract(bops.contract(
            traced_node_proj_top, double_leg_projector, '0', '0'), double_leg_projector, '0', '0'), double_leg_projector, '0', '0'), double_leg_projector, '0', '0')
        traced_node_proj_right = tn.Node(bops.contract(bops.contract(node, proj_1, '5', '0'),
                                node, '45', '45*').tensor.transpose([0, 4, 1, 5, 2, 6, 3, 7]).reshape([4**2] * 4))
        traced_node_proj_right = bops.contract(bops.contract(bops.contract(bops.contract(
            traced_node_proj_right, double_leg_projector, '0', '0'), double_leg_projector, '0', '0'), double_leg_projector, '0', '0'), double_leg_projector, '0', '0')
        traced_node_proj_corner = tn.Node(bops.contract(bops.contract(bops.contract(node, proj_1, '4', '0'), proj_1, '4', '0'),
                                node, '45', '45*').tensor.transpose([0, 4, 1, 5, 2, 6, 3, 7]).reshape([4**2] * 4))
        traced_node_proj_corner = bops.contract(bops.contract(bops.contract(bops.contract(
            traced_node_proj_corner, double_leg_projector, '0', '0'), double_leg_projector, '0', '0'), double_leg_projector, '0', '0'), double_leg_projector, '0', '0')
        if bc == 'P':
            norm = get_expectation_value(N, [[traced_node_full_projected] * N] * N)
            if N < 4:
                plaquette_term = get_expectation_value(N, [[traced_node_x_right, traced_node_outgoing_projected] + [traced_node_full_projected] * (N-2),
                                                           [traced_node_xx, traced_node_x_top] + [traced_node_full_projected] * (N-2)]
                                                         + [[traced_node_full_projected] * N] * (N-2))
            elif N == 4:
                top_right = tn.Node(bops.contract(bops.contract(traced_node_x_right, traced_node_outgoing_projected, '1', '3'),
                    bops.contract(traced_node_xx, traced_node_x_top, '1', '3'), '15', '03')\
                    .tensor.transpose([0, 2, 3, 6, 4, 7, 1, 5]).reshape([traced_node_x_right[0].dimension**2] * 4))
                env = tn.Node(bops.contract(bops.contract(traced_node_full_projected, traced_node_full_projected, '1', '3'),
                                          bops.contract(traced_node_full_projected, traced_node_full_projected, '1', '3'), '15', '03') \
                    .tensor.transpose([0, 2, 3, 6, 4, 7, 1, 5]).reshape([traced_node_x_right[0].dimension ** 2] * 4))
                plaquette_term = get_expectation_value(2, [[top_right, env], [env, env]])
            magnetic_term = get_expectation_value(N, [[traced_node_z_top] + [traced_node_full_projected] * (N-1)] + [[traced_node_full_projected] * N] * (N-1))
            small_loop_term = get_expectation_value(N, [[traced_node_proj_right] + [traced_node_full_projected] * (N-1),
                                                       [traced_node_proj_corner, traced_node_proj_top] + [traced_node_full_projected] * (N-2)]
                                                     + [[traced_node_full_projected] * N] * (N-2))
            return -Jp * N**2 * plaquette_term / norm - hz * 2 * N**2 * magnetic_term / norm + N**2 * small_loop_c * small_loop_term / norm
        else: # bc == 'O':
            edge = tn.Node(np.array([1] + [0] * (traced_node_full_projected[0].dimension - 1)))
            norm = get_expectation_value(N, [[traced_node_full_projected] * (N+1)] * (N+1), bc=bc, edge=edge)
            if N % 2 == 1:
                plaquette_term = 0
                plaquette_positions = [[i, j] for i in range(N//2 + 1) for j in range(N//2 + 1) if i <= j]
                for pi in range(len(plaquette_positions)):
                    position = plaquette_positions[pi]
                    nodes = [[traced_node_full_projected] * (N+1)] * position[0] \
                            + [[traced_node_full_projected] * position[1] + [traced_node_x_right] + [traced_node_outgoing_projected] + [traced_node_full_projected] * (N + 1 - 2 - position[1])] \
                            + [[traced_node_full_projected] * position[1] + [traced_node_xx] + [traced_node_x_top] + [traced_node_full_projected] * (N + 1 - 2 - position[1])] \
                            + [[traced_node_full_projected] * (N+1)] * (N + 1 - 2 - position[0])
                    plaquette_term += - get_expectation_value(N, nodes, bc=bc, edge=edge) * Jp * 4 * (1 + int(position[0] == position[1] or position[1] == (N//2 + 1)))
                z_positions = [[i, j] for i in range(N//2 + 1) for j in range(N//2 + 1)]
                z_term = 0
                for pi in range(len(z_positions)):
                    position = z_positions[pi]
                    nodes = [[traced_node_full_projected] * (N+1)] * position[0] \
                            + [[traced_node_full_projected] * position[1] + [traced_node_z_right] + [traced_node_full_projected] * (N - position[1])] \
                            + [[traced_node_full_projected] * (N+1)] * (N - position[0])
                    z_term += -hz * get_expectation_value(N, nodes, bc=bc, edge=edge) * 4 * (1 + int(position[1] < (N//2 + 1)))
                # TODO handle small loop term
                return (plaquette_term + z_term) / norm



def infinite_system_energy(params, Jp: float, hz: float, is_sr=True):
    node = minimal_ansatz_node(params=params, is_sr=is_sr)
    d = 2
    I, X, Z = np.eye(d), np.array([[0, 1], [1, 0]]), np.diag([1, -1])
    node_env = tn.Node(bops.contract(node, node, '4', '4*')
                       .tensor.transpose([0, 4, 1, 5, 2, 6, 3, 7]).reshape([node[0].dimension ** 2] * 4))
    node_env.tensor /= np.sqrt(bops.contract(node, node, '01234', '01234*').tensor)
    upRow, downRow, leftRow, rightRow = peps.applyBMPS(node_env, node_env, gauge=True)
    [cUp, dUp, te] = bops.svdTruncation(upRow, [0, 1], [2, 3], '>>', normalize=True)
    [cDown, dDown, te] = bops.svdTruncation(downRow, [0, 1], [2, 3], '>>', normalize=True)

    norm = pe.applyLocalOperators(cUp, dUp, cDown, dDown, leftRow, rightRow, node, node, 2, 2,
                                  [tn.Node(np.eye(d ** 4))] * 4)
    plaquette_ops = [tn.Node(np.kron(I, np.kron(X, np.kron(X, I)))), tn.Node(np.eye(d**4)),
                     tn.Node(np.kron(X, np.kron(I, np.kron(I, X)))), tn.Node(np.eye(d**4))]
    plaquette_term = pe.applyLocalOperators(cUp, dUp, cDown, dDown, leftRow, rightRow, node, node, 2, 2,
                                            plaquette_ops)
    h_ops = [tn.Node(np.kron(Z, np.eye(d**3))), tn.Node(np.eye(d**4)), tn.Node(np.eye(d**4)), tn.Node(np.eye(d**4))]
    h_term = pe.applyLocalOperators(cUp, dUp, cDown, dDown, leftRow, rightRow, node, node, 2, 2, h_ops)
    result = -Jp * plaquette_term / (norm * 2) - hz * h_term / norm
    # print(params, 'energy', result, 'plaquette', plaquette_term, 'h', h_term)
    return result


def gradient_descent(learning_rate, params, Jp: float, hz: float, is_sr=True, accuracy=1e-2):
    E = infinite_system_energy(params, Jp=Jp, hz=hz, is_sr=is_sr)
    E_ds = np.array([1] * len(params), dtype=float)
    steps = [learning_rate] * len(params)
    for si in range(10000):
        new_E_ds = [0] * len(params)
        no_sr_inds = 2
        for i in range(no_sr_inds):
            E_ds[i] = (infinite_system_energy(params[:i] + [params[i] + steps[i]] + params[i+1:], Jp, hz, is_sr=is_sr) \
                       - infinite_system_energy(params[:i] + [params[i] - steps[i]] + params[i+1:], Jp, hz, is_sr=is_sr)) / (2 * steps[i])
            params[i] -= steps[i] * E_ds[i]
        # if np.abs(new_E_d_alpha) > np.abs(E_d_alpha): # si % 20 == 0: #
        #     alpha_step /= 2
        # if np.abs(new_E_d_beta) > np.abs(E_d_beta): # si % 20 == 0: #
        #     print(beta_step)
        #     beta_step /= 2
        if is_sr:
            for i in range(no_sr_inds, len(params)):
                E_ds[i] = (infinite_system_energy(params[:i] + [params[i] + steps[i]] + params[i+1:], Jp, hz, is_sr=is_sr) - E) / steps[i]
                params[i] -= steps[i] * E_ds[i]
        E_new = infinite_system_energy(params, Jp, hz, is_sr=is_sr)
        if E < E_new:
            dbg = 1
        print(E, E_new, np.abs(E_ds))
        if max(np.abs(E_ds[:no_sr_inds])) < accuracy:
            if not is_sr:
                print(E, E_new, np.abs(E_ds))
                break
            else:
                if max(np.abs(E_ds[no_sr_inds:])) < accuracy:
                    print(E, E_new, np.abs(E_ds))
                    break
        E = E_new
    return E, params

lambdas = [0.2 * i for i in range(int(sys.argv[1]), int(sys.argv[2]))] # + [np.round(0.05 * i, 10) for i in range(1, 4)]
E0s = np.zeros(len(lambdas))
E1s = np.zeros(len(lambdas))
N = 2 #int(sys.argv[1])
small_loop_c = 0.1
bc = 'O'
dirname = sys.argv[3]
for li in range(len(lambdas)):
    print(li)
    filename = dirname + 'no_sr_N_' + str(N) + '_lambda_' + str(lambdas[li]) + '_bc_' + bc
    if not os.path.exists(filename):
        E_0_order, params_0 = gradient_descent(1e-2, [0.1] + [0.5] * 4, 1 / 2 / lambdas[li], lambdas[li] / 2, is_sr=False)
        print('-------')
        E_1_order, params_1 = gradient_descent(1e-2, params_0[:2] + [0] * 3, 1 / 2 / lambdas[li], lambdas[li] / 2, is_sr=True)
        pickle.dump([E_0_order, params_0, E_1_order, params_1], open(filename, 'wb'))
    else:
        # E_0_order, alpha_0, beta_0, c, E_1_order, alpha_, beta_1, c_1 = pickle.load(open(filename, 'rb'))
        # print([alpha_0, beta_0, c], [alpha_, beta_1, c_1])
        E_0_order, params_0, E_1_order, params_1 = pickle.load(open(filename, 'rb'))
        print(E_0_order, params_0, E_1_order, params_1)
    E0s[li] = E_0_order
    E1s[li] = E_1_order
plot = True
if plot:
    import matplotlib.pyplot as plt
    plt.scatter(lambdas, E0s, marker='+')
    # plt.plot(lambdas, Es_ansatz)
    plt.scatter(lambdas, E1s, marker='+')
    plt.show()
    plt.plot(lambdas, (E0s - E1s)/E1s)
    plt.show()
