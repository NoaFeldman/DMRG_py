import tensornetwork as tn
import numpy as np
import math
from typing import Any, Dict, List, Optional, Set, Text, Tuple, Union, \
    Sequence, Iterable, Type
# import jax.numpy as jnp
# import torch

def torchTranspose(arr, dim):
    return arr.permute(dim)

# BACKEND = 'numpy' # tensorflow, pytorch, numpy, symmetric
def init(backend, dev=None):
    global BACKEND
    global transpose
    global reshape
    global conj
    global device
    global dtype
    BACKEND = backend
    tn.set_default_backend(BACKEND)
    if BACKEND == 'numpy':
        transpose = np.transpose
        reshape = np.reshape
        conj = np.conj
    elif BACKEND == 'jax':
        transpose = jnp.transpose
        reshape = jnp.reshape
        conj = jnp.conj
    elif BACKEND == 'pytorch':
        transpose = torchTranspose
        reshape = torch.reshape
        conj = torch.conj
        device = torch.device(dev)
        dtype = torch.float


init('numpy')
def setBackend(backend, dev=None):
    init(backend, dev)

def getLegsSplitterTensor(dims):
    fullDim = np.product(dims)
    splitter = np.eye(fullDim, dtype=complex).reshape(dims + [fullDim])
    return splitter


# Assumes unified legs are in consecutive order
def unifyLegs(node: tn.Node, legs: List[int], cleanOriginal=True) -> tn.Node:
    shape = node.get_tensor().shape
    newTensor = reshape(node.get_tensor(), list(shape[:legs[0]]) + [np.product([shape[i] for i in legs])] +
                        list(shape[legs[-1] + 1:]))
    if cleanOriginal:
        tn.remove_node(node)
    return tn.Node(newTensor)


def getStartupState(n, d=2, mode='general'):
    psi = [None] * n
    if mode == 'general':
        if d == 2:
            baseLeftTensor = np.zeros((1, 2, 2), dtype=complex)
            baseLeftTensor[0, 1, 0] = -1
            baseLeftTensor[0, 0, 1] = 1
        elif d == 3:
            print("TODO")
        psi[0] = tn.Node(baseLeftTensor, name='site0', axis_names=['v0', 's0', 'v1'], backend=BACKEND)
        if d == 2:
            baseMiddleTensor = np.zeros((2, 2, 2), dtype=complex)
            baseMiddleTensor[0, 1, 0] = -1 / math.sqrt(2)
            baseMiddleTensor[1, 0, 1] = 1 / math.sqrt(2)
            baseMiddleTensor[1, 1, 0] = -1 / math.sqrt(2)
            baseMiddleTensor[0, 0, 1] = 1 / math.sqrt(2)
        elif d == 3:
            print("TODO")
        for i in range(1, n-1):
            psi[i] = tn.Node(baseMiddleTensor, name=('site' + str(i)),
                                   axis_names=['v' + str(i), 's' + str(i), 'v' + str(i+1)],
                                   backend=BACKEND)
        if d == 2:
            baseRightTensor = np.zeros((2, 2, 1), dtype=complex)
            baseRightTensor[0, 1, 0] = -1 / math.sqrt(2)
            baseRightTensor[1, 0, 0] = 1 / math.sqrt(2)
        elif d == 3:
            print("TODO")
        psi[n - 1] = tn.Node(baseRightTensor, name=('site' + str(n - 1)),
                                   axis_names=['v' + str(n - 1), 's' + str(n - 1), 'v' + str(n)],
                                   backend=BACKEND)
        norm = getOverlap(psi, psi)
        psi[n-1] = multNode(psi[n-1], 1/np.sqrt(norm))
        return psi
    elif mode == 'antiferromagnetic':
        if d == 2:
            baseLeftTensor = np.zeros((1, 2, 2), dtype=complex)
            baseLeftTensor[0, 0, 0] = 1
            baseLeftTensor[0, 1, 1] = 1
        elif d == 3:
            baseLeftTensor = np.zeros((1, 3, 1), dtype=complex)
            baseLeftTensor[0, 0, 0] = 1
        psi[0] = tn.Node(baseLeftTensor, name='site0', axis_names=['v0', 's0', 'v1'],
                         backend=BACKEND)
        if d == 2:
            baseMiddleTensorEven = np.zeros((2, 2, 2), dtype=complex)
            baseMiddleTensorEven[0, 0, 0] = 1
            baseMiddleTensorEven[1, 1, 1] = 1
            baseMiddleTensorOdd = np.zeros((2, 2, 2), dtype=complex)
            baseMiddleTensorOdd[0, 1, 0] = 1
            baseMiddleTensorOdd[1, 0, 1] = 1
        elif d == 3:
            baseMiddleTensorEven = np.zeros((1, 3, 1), dtype=complex)
            baseMiddleTensorEven[0, 0, 0] = 1
            baseMiddleTensorOdd = np.zeros((1, 3, 1), dtype=complex)
            baseMiddleTensorOdd[0, 2, 0] = 1
        for i in range(1, n - 1):
            if i % 2 == 0:
                psi[i] = tn.Node(baseMiddleTensorEven, name=('site' + str(i)),
                             axis_names=['v' + str(i), 's' + str(i), 'v' + str(i + 1)],
                             backend=BACKEND)
            else:
                psi[i] = tn.Node(baseMiddleTensorOdd, name=('site' + str(i)),
                                 axis_names=['v' + str(i), 's' + str(i), 'v' + str(i + 1)],
                                 backend=BACKEND)
        if d == 2:
            baseRightTensor = np.zeros((2, 2, 1), dtype=complex)
            baseRightTensor[0, 1, 0] = 1
            baseRightTensor[1, 0, 0] = 1
        elif d == 3:
            baseRightTensor = np.zeros((1, 3, 1), dtype=complex)
            baseRightTensor[0, 2, 0] = 1
        psi[n - 1] = tn.Node(baseRightTensor, name=('site' + str(n - 1)),
                             axis_names=['v' + str(n - 1), 's' + str(n - 1), 'v' + str(n)],
                             backend=BACKEND)
        norm = getOverlap(psi, psi)
        psi[n - 1] = multNode(psi[n - 1], 1 / np.sqrt(norm))
        return psi
    elif mode == 'aklt':
        baseTensor = np.zeros((2, 3, 2), dtype=complex)
        baseTensor[0, 0, 1] = np.sqrt(2 / 3)
        baseTensor[0, 1, 0] = -np.sqrt(1 / 3)
        baseTensor[1, 1, 1] = np.sqrt(1 / 3)
        baseTensor[1, 2, 0] = -np.sqrt(2 / 3)
        leftTensor = np.zeros((1, 3, 2), dtype=complex)
        leftTensor[0, 0, 0] = 1
        leftTensor[0, 1, 1] = 1 / np.sqrt(2)
        leftTensor[0, 2, 1] = 1 / np.sqrt(2)
        rightTensor = np.zeros((2, 3, 1), dtype=complex)
        rightTensor[0, 0, 0] = 1
        rightTensor[1, 1, 0] = 1
        rightTensor[1, 2, 0] = 1
        psi[0] = tn.Node(leftTensor, backend=BACKEND)
        psi[n-1] = tn.Node(rightTensor, backend=BACKEND)
        for i in range(1, n-1):
            psi[i] = tn.Node(baseTensor, name=('site' + str(i)),
                                 axis_names=['v' + str(i), 's' + str(i), 'v' + str(i + 1)],
                                 backend=BACKEND)

        norm = getOverlap(psi, psi)
        psi[n - 1] = multNode(psi[n - 1], 1 / np.sqrt(norm))
        return psi


# Assuming psi1, psi2 have the same length, Hilbert space etc.
# assuming psi2 is conjugated
def getOverlap(psi1Orig: List[tn.Node], psi2Orig: List[tn.Node]):
    if len(psi1Orig) == 1:
        return multiContraction(psi1Orig[0], psi2Orig[0], '012', '012*').tensor
    psi1 = copyState(psi1Orig)
    psi2 = copyState(psi2Orig, conj=True)
    psi1[0][0] ^ psi2[0][0]
    psi1[0][1] ^ psi2[0][1]
    contracted = tn.contract_between(psi1[0], psi2[0], name='contracted')
    for i in range(1, len(psi1) - 1):
        psi1[i][1] ^ psi2[i][1]
        contracted[0] ^ psi1[i][0]
        contracted[1] ^ psi2[i][0]
        contracted = tn.contract_between(tn.contract_between(contracted, psi1[i]), psi2[i])
    psi1[len(psi1) - 1][1] ^ psi2[len(psi1) - 1][1]
    psi1[len(psi1) - 1][2] ^ psi2[len(psi1) - 1][2]
    contracted[0] ^ psi1[len(psi1) - 1][0]
    contracted[1] ^ psi2[len(psi1) - 1][0]
    contracted = tn.contract_between(tn.contract_between(contracted, psi1[len(psi1) - 1]), psi2[len(psi1) - 1])

    result = contracted.tensor
    tn.remove_node(contracted)
    removeState(psi1)
    removeState(psi2)
    return result

def printNode(node):
    if node == None:
        print('None')
        return
    print('node ' + node.name + ':')
    edgesNames = ''
    for edge in node.edges:
        edgesNames += edge.name + ', '
    print(edgesNames)
    print(node.tensor.shape)


def copyState(psi, conj=False) -> List[tn.Node]:
    result = list(tn.copy(psi, conjugate=conj)[0].values())
    if conj:
        for node in result:
            for edge in node.edges:
                if edge.name[len(edge.name) - 1] == '*':
                    edge.name = edge.name[0:len(edge.name) - 1]
                else:
                    edge.name = edge.name + '*'
    return result

def addNodes(node1, node2, cleanOr1=False, cleanOr2=False):
    # TODO asserts
    if node1 is None:
        if node2 is None:
            res = None
        else:
            res = node2
    else:
        if node2 is None:
            res = node1
        else:
            result = copyState([node1])[0]
            result.set_tensor(result.get_tensor() + node2.get_tensor())
            res = result
    if cleanOr1:
        tn.remove_node(node1)
    if cleanOr2:
        tn.remove_node(node2)
    return res

def multNode(node: tn.Node, c):
    node.set_tensor(node.get_tensor() * c)
    return node


def getNodeNorm(node: tn.Node):
    copy = copyState([node])[0]
    copyConj = copyState([node], conj=True)[0]
    for i in range(node.get_rank()):
        copy[i] ^ copyConj[i]
    norm2 = tn.contract_between(copy, copyConj).get_tensor()
    if BACKEND == 'pytorch':
        norm2 = norm2.numpy()
    return np.sqrt(norm2)


def contract(node1: tn.Node, node2: tn.Node, edges1, edges2, nodeName=None,
                     cleanOr1=False, cleanOr2=False, isDiag1=False, isDiag2=False) -> tn.Node:
    return multiContraction(node1, node2, edges1, edges2, nodeName,
                            cleanOr1, cleanOr2, isDiag1, isDiag2)


def multiContraction(node1: tn.Node, node2: tn.Node, edges1, edges2, nodeName=None,
                     cleanOr1=False, cleanOr2=False, isDiag1=False, isDiag2=False) -> tn.Node:
    if node1 is None or node2 is None:
        return None
    if edges1[len(edges1) - 1] == '*':
        copy1 = copyState([node1], conj=True)[0]
        edges1 = edges1[0:len(edges1) - 1]
    else:
        copy1 = copyState([node1])[0]
    if edges2[len(edges2) - 1] == '*':
        copy2 = copyState([node2], conj=True)[0]
        edges2 = edges2[0:len(edges2) - 1]
    else:
        copy2 = copyState([node2])[0]

    if cleanOr1:
        tn.remove_node(node1)
    if cleanOr2:
        tn.remove_node(node2)
    if isDiag1 and isDiag2:
        return tn.Node(copy1.tensor * copy2.tensor)
    elif isDiag1 and not isDiag2:
        return contractDiag(copy2, copy1.tensor, int(edges2[0]))
    elif isDiag2 and not isDiag1:
        return contractDiag(copy1, copy2.tensor, int(edges1[0]))
    for i in range(len(edges1)):
        copy1[int(edges1[i])] ^ copy2[int(edges2[i])]
    return tn.contract_between(copy1, copy2, name=nodeName)


def contractDiag(node: tn.Node, diag: np.array, edgeNum: int):
    node.tensor = transpose(node.tensor,
                               [edgeNum] + [i for i in range(len(node.edges)) if i != edgeNum])
    for i in range(node[0].dimension):
        node.tensor[i] *= diag[i]
    node.tensor = transpose(node.tensor,
                               list(range(1, edgeNum + 1)) + [0] + list(range(edgeNum + 1, len(node.edges))))
    return node




def permute(node: tn.Node, perm: List[int]) -> tn.Node:
    node.tensor = node.tensor.transpose(perm)
    return node


def dagger(mat:np.array) -> np.array:
    return np.conj(np.transpose(mat))


def inner_contract(node: tn.Node, indices: List[int]) -> tn.Node:
    id = tn.Node(np.eye(node.tensor.shape[indices[0]]))
    return contract(node, id, indices, '01')


def svdTruncation(node: tn.Node, leftEdges: List[int], rightEdges: List[int],
                  dir: str, maxBondDim=1024, leftName='U', rightName='V',  edgeName='default', normalize=False, maxTrunc=15):
    # np.seterr(all='raise')
    maxBondDim = getAppropriateMaxBondDim(maxBondDim,
                                          [node.edges[e] for e in leftEdges], [node.edges[e] for e in rightEdges])
    if dir == '>>':
        leftEdgeName = edgeName
        rightEdgeName = None
    else:
        leftEdgeName = None
        rightEdgeName = edgeName
    try:
        [U, S, V, truncErr] = tn.split_node_full_svd(node, [node.edges[e] for e in leftEdges],
                                                 [node.edges[e] for e in rightEdges], max_singular_values=maxBondDim,
                                       left_name=leftName, right_name=rightName,
                                       left_edge_name=leftEdgeName, right_edge_name=rightEdgeName)

    except np.linalg.LinAlgError:
        # TODO
        b = 1
        node.tensor = np.round(node.tensor, 16)
        [U, S, V, truncErr] = tn.split_node_full_svd(node, [node.edges[e] for e in leftEdges],
                                                     [node.edges[e] for e in rightEdges],
                                                     max_singular_values=maxBondDim,
                                                     left_name=leftName, right_name=rightName,
                                                     left_edge_name=leftEdgeName, right_edge_name=rightEdgeName)
    # s = S
    # S = tn.Node(np.diag(S.tensor))
    # tn.remove_node(s)
    # norm = np.sqrt(sum(S.tensor**2))
    norm = np.sqrt(sum(np.diag(S.tensor**2)))
    if norm == 0:
        b = 1
    if maxTrunc > 0:
        meaningful = sum(np.round(np.diag(S.tensor) / norm, maxTrunc) > 0)
        S.tensor = S.tensor[:meaningful, :meaningful]
        U.tensor = U.tensor.transpose([-1 * i for i in range(1, len(U.tensor.shape) + 1)])[:meaningful, :]. \
            transpose([-1 * i for i in range(1, len(U.tensor.shape) + 1)])
        # U.tensor = U.tensor[:, :, :meaningful]
        V.tensor = V.tensor[:meaningful, :]
    if normalize:
        S = multNode(S, 1 / norm)
    for e in S.edges:
        e.name = edgeName
    if dir == '>>':
        l = copyState([U])[0]
        # r = multiContraction(S, V, '1', '0', cleanOr1=True, cleanOr2=True, isDiag1=True)
        r = multiContraction(S, V, '1', '0', cleanOr1=True, cleanOr2=True)
    elif dir == '<<':
        # l = multiContraction(U, S, [len(U.edges) - 1], '0', cleanOr1=True, cleanOr2=True, isDiag2=True)
        l = multiContraction(U, S, [len(U.edges) - 1], '0', cleanOr1=True, cleanOr2=True)
        r = copyState([V])[0]
    elif dir == '>*<':
        v = V
        V = copyState([V])[0]
        tn.remove_node(v)
        u = U
        U = copyState([U])[0]
        tn.remove_node(u)
        return [U, S, V, truncErr]

    tn.remove_node(U)
    tn.remove_node(S)
    tn.remove_node(V)
    return [l, r, truncErr]


def getRenyiEntropy(psi: List[tn.Node], n: int, ASize: int, maxBondDim=1024):
    psiCopy = copyState(psi)
    for k in [len(psiCopy) - 1 - i for i in range(len(psiCopy) - ASize - 1)]:
        psiCopy = shiftWorkingSite(psiCopy, k, '<<')
    M = multiContraction(psiCopy[ASize - 1], psiCopy[ASize], [2], [0])

    leftEdges = M.edges[:2]
    rightEdges = M.edges[2:]
    maxBondDim = getAppropriateMaxBondDim(maxBondDim, leftEdges, rightEdges)

    [U, S, V, truncErr] = tn.split_node_full_svd(M, leftEdges, rightEdges, max_singular_values=maxBondDim)
    eigenvaluesRoots = np.diag(S.tensor)
    result = sum([l ** (2 * n) for l in eigenvaluesRoots])
    removeState(psiCopy)
    return result


# Apparently the truncation method doesn't like it if max_singular_values is larger than the size of S.
def getAppropriateMaxBondDim(maxBondDim, leftEdges, rightEdges):
    uDim = 1
    for e in leftEdges:
        uDim *= e.dimension
    vDim = 1
    for e in rightEdges:
        vDim *= e.dimension
    if maxBondDim > min(uDim, vDim):
        return min(uDim, vDim)
    else:
        return maxBondDim


# Split M into 2 3-rank tensors for sites k, k+1
def assignNewSiteTensors(psi, k, M, dir, getOrthogonal=False, maxBondDim=128):
    [sitek, sitekPlus1, truncErr] = svdTruncation(M, [0, 1], [2, 3], dir, \
            leftName=('site' + str(k)), rightName=('site' + str(k+1)), edgeName = ('v' + str(k+1)), maxBondDim=maxBondDim)
    tn.remove_node(psi[k])
    psi[k] = sitek
    # if k > 0:
    #     psi[k-1][2] ^ psi[k]
    tn.remove_node(psi[k+1])
    psi[k+1] = sitekPlus1
    # if k+2 < len(psi):
    #     psi[k+1][2] ^ psi[k+2][0]
    if BACKEND == 'pytorch':
        truncErr = truncErr.numpy()
    return [psi, truncErr]


def getEdgeNames(node: tn.Node):
    result = []
    for edge in node.edges:
        result.append(edge.name)
    return result


# k is curr working site, shift it by one in dir direction.
def shiftWorkingSite(psi: List[tn.Node], k, dir, maxBondDim=1024):
    if dir == '<<':
        pair = multiContraction(psi[k-1], psi[k], [2], [0], cleanOr1=True, cleanOr2=True)
        if maxBondDim is None:
            [l, r, I] = svdTruncation(pair, [0, 1], [2, 3], '<<')
        else:
            [l, r, I] = svdTruncation(pair, [0, 1], [2, 3], '<<', maxBondDim=maxBondDim)
        psi[k - 1] = l
        psi[k] = r
        tn.remove_node(pair)
    else:
        pair = tn.contract(psi[k][2] ^ psi[k+1][0])
        [l, r, I] = svdTruncation(pair, [0, 1], [2, 3], '>>', maxBondDim=maxBondDim)
        psi[k] = l
        psi[k + 1] = r
        tn.remove_node(pair)
    return psi


def moveOrthogonalityCenter(psi, n):
    k = len(psi) - 1
    while k > n:
        psi = shiftWorkingSite(psi, k, '<<')
        k -=1
    return psi


def removeState(psi):
    for i in range(len(psi)):
        tn.remove_node(psi[i])


def addStates(psi1: List[tn.Node], psi2: List[tn.Node]):
    result = copyState(psi1)
    resultTensor = np.zeros((1, psi1[0].shape[1], psi1[0].shape[2] + psi2[0].shape[2]), dtype=complex)
    resultTensor[0, :, :psi1[0].shape[2]] = psi1[0].tensor
    resultTensor[0, :, psi1[0].shape[2]:] = psi2[0].tensor
    result[0].set_tensor(resultTensor)
    for i in range(1, len(psi1)-1):
        resultTensor = \
            np.zeros((psi1[i].shape[0] + psi2[i].shape[0], psi1[i].shape[1], psi1[i].shape[2] + psi2[i].shape[2]), dtype=complex)
        resultTensor[:psi1[i].shape[0], :, :psi1[i].shape[2]] = psi1[i].tensor
        resultTensor[psi1[i].shape[0]:, :, psi1[i].shape[2]:] = psi2[i].tensor
        result[i].set_tensor(resultTensor)
    resultTensor = np.zeros((psi1[len(psi1)-1].shape[0] + psi2[len(psi1)-1].shape[0], psi1[len(psi1)-1].shape[1], 1), dtype=complex)
    resultTensor[:psi1[len(psi1)-1].shape[0], :, :] = psi1[len(psi1)-1].tensor
    resultTensor[psi1[len(psi1)-1].shape[0]:, :, :] = psi2[len(psi1)-1].tensor
    result[len(psi1)-1].set_tensor(resultTensor)
    return result


def getOrthogonalState(psi: List[tn.Node], psiInitial=None):
    psiCopy = copyState(psi)
    if psiInitial is None:
        psiInitial = getStartupState(len(psi))
    overlap = getOverlap(psiCopy, psiInitial)
    psiCopy[0] = multNode(psiCopy[0], -overlap)
    result = addStates(psiInitial, psiCopy)
    result[len(result)-1] = multNode(result[len(result)-1], 1/math.sqrt(getOverlap(result, result)))
    removeState(psiCopy)
    k = len(result)-1
    while k > 0:
        result = shiftWorkingSite(result, k, '<<')
        k -= 1
    while k < len(result)-1:
        result = shiftWorkingSite(result, k, '>>')
        k += 1
    return result


# Assuming psi's working site is the last from the left
def normalize(psi: List[tn.Node]):
    norm = math.sqrt(getOverlap(psi, psi))
    psi[len(psi)-1] = multNode(psi[len(psi)-1], 1 / norm)
    return psi


def minusState(psi: List[tn.Node]):
    psiCopy = copyState(psi)
    psiCopy[0] = multNode(psiCopy[0], -1)
    return psiCopy


def applySingleSiteOp(psi: List[tn.Node], op: tn.Node, i: int):
    psi[i][1] ^ op[1]
    psi[i] = tn.contract_between(psi[i], op).reorder_axes([0, 2, 1])


# n spins, last two are maximally entangled and the first n - 2 are in a product state with them.
def getTestState_pair(n):
    psi = [None] * n
    leftTensor = np.ones((1, 2, 1), dtype=complex)
    leftTensor *= 1 / np.sqrt(2)
    for i in range(n-2):
        psi[i] = tn.Node(leftTensor)
    middleTensor = np.zeros((1, 2, 2), dtype=complex)
    middleTensor[0, 0, 0] = 1
    middleTensor[0, 1, 1] = 1
    psi[n-2] = tn.Node(middleTensor)
    rightTensor = np.zeros((2, 2, 1))
    rightTensor[0, 0, 0] = 1 / np.sqrt(2)
    rightTensor[1, 1, 0] = 1 / np.sqrt(2)
    psi[n-1] = tn.Node(rightTensor)
    norm = getOverlap(psi, psi)
    psi[n - 1] = multNode(psi[n - 1], 1 / np.sqrt(norm))
    return psi

# both halves of the systems are in the maximally entangled state of a pair (0000000 + 111111)
def getTestState_halvesAsPair(n):
    psi = [None] * n
    leftTensor = np.zeros((1, 2, 2), dtype=complex)
    leftTensor[0, 0, 0] = 1
    leftTensor[0, 1, 1] = 1
    psi[0] = tn.Node(leftTensor)
    midTensor = np.zeros((2, 2, 2), dtype=complex)
    midTensor[0, 0, 0] = 1
    midTensor[1, 1, 1] = 1
    for i in range(1, n-1):
        psi[i] = tn.Node(midTensor)
    rightTensor = np.zeros((2, 2, 1), dtype=complex)
    rightTensor[0, 0, 0] = 1
    rightTensor[1, 1, 0] = 1
    psi[n-1] = tn.Node(rightTensor)
    norm = getOverlap(psi, psi)
    psi[int(n / 2) - 1] = multNode(psi[int(n / 2) - 1], 1 / np.sqrt(norm))
    for k in range(int(n / 2) - 1, n - 1):
        psi = shiftWorkingSite(psi, k, '>>')
    return psi

# Both halves of the state are maximally entangled
# working site is middle site
def getTestState_maximallyEntangledHalves(n):
    psi = [None] * n
    for i in range(int(n / 2)):
        tensor = np.zeros((2**i, 2, 2**(i+1)), dtype=complex)
        for j in range(2**i):
            tensor[j, 0, 2 * j] = 1
            tensor[j, 1, 2 * j + 1] = 1
        psi[i] = tn.Node(tensor)
    for i in range(int(n / 2), n):
        tensor = np.zeros((2**(n - i), 2, 2**(n - 1 - i)), dtype=complex)
        for j in range(2 ** (n - i)):
            currBit = int((2 ** (n - i - 1) & j) > 0)
            tensor[j, currBit, j % 2 ** (n - i - 1)] = 1
        psi[i] = tn.Node(tensor)
    norm = getOverlap(psi, psi)
    psi[int(n/2)] = multNode(psi[int(n/2)], 1 / np.sqrt(norm))
    for k in range(int(n/2), n - 1):
        psi = shiftWorkingSite(psi, k, '>>')
    return psi

def getTestState_unequalTwoStates(n, weight0):
    psi = getTestState_halvesAsPair(n)
    psi[n - 1].tensor[0, 0, 0] = weight0
    psi[n - 1].tensor[1, 1, 0] = np.sqrt(1 - weight0**2)
    return psi


def relaxState(psi, maxBondDim):
    psiCopy = copyState(psi)
    for k in range(len(psi) - 1, 1, -1):
        psiCopy = shiftWorkingSite(psiCopy, k, '<<', maxBondDim=maxBondDim)
    for k in range(1, len(psi) - 1):
        psiCopy = shiftWorkingSite(psiCopy, k, '>>', maxBondDim=maxBondDim)
    return psiCopy


def getExplicitDM(psi, N, d=2):
    curr = psi[0]
    for i in range(1, N):
        curr = multiContraction(curr, psi[i], [i+1], '0')
    dm = reshape(multiContraction(curr, curr,
            [0, len(curr.edges) - 1], [0, len(curr.edges) - 1, '*']).tensor, [d**N, d**N])
    return dm


def getExplicitVec(psi, d=2):
    curr = psi[0]
    for i in range(1, len(psi)):
        curr = multiContraction(curr, psi[i], [i + 1], '0')
    return reshape(curr.tensor, [d**len(psi)])


def getExpectationValue(psi: List[tn.Node], ops: List[tn.Node], opt='single_site_ops'):
    psiCopy = copyState(psi)
    for i in range(len(psiCopy)):
        psiCopy[i] = multiContraction(psiCopy[i], ops[i], '1', '1').reorder_axes([0, 2, 1])
        if opt == 'multi_site_ops':
            psiCopy[i] = unifyLegs(unifyLegs(permute(psiCopy[i], [0, 3, 2, 1, 4]), [3, 4]), [0, 1])
    result = getOverlap(psi, psiCopy)
    removeState(psiCopy)
    return result


def vector_to_mps(psi, d, N):
    orig = tn.Node(psi.reshape([1] + [d] * N + [1]))
    result = []
    for i in range(N - 1):
        [new_site, orig, te] = svdTruncation(orig, [0, 1], list(range(2, len(orig.tensor.shape))), '>>')
        result.append(new_site)
    result.append(orig)
    return result