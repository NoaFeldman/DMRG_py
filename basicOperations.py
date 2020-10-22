import tensornetwork as tn
import tensornetwork.backends.base_backend as be
import numpy as np
import math
from typing import Any, Dict, List, Optional, Set, Text, Tuple, Union, \
    Sequence, Iterable, Type


def getLegsSplitterTensor(dim1, dim2):
    splitter = np.zeros((dim1, dim2, dim1 * dim2), dtype=complex)
    for i in range(dim1):
        for j in range(dim2):
            splitter[i, j, i * dim2 + j] = 1
    return splitter


# Assumes unified legs are in consecutive order
def unifyLegs(node: tn.Node, leg1: int, leg2: int, cleanOriginal=True) -> tn.Node:
    shape = node.get_tensor().shape
    newTensor = np.reshape(node.get_tensor(), list(shape[:leg1]) + [shape[leg1] * shape[leg2]] + list(shape[leg2 + 1:]))
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
        psi[0] = tn.Node(baseLeftTensor, name='site0', axis_names=['v0', 's0', 'v1'], backend=None)
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
                                   backend=None)
        if d == 2:
            baseRightTensor = np.zeros((2, 2, 1), dtype=complex)
            baseRightTensor[0, 1, 0] = -1 / math.sqrt(2)
            baseRightTensor[1, 0, 0] = 1 / math.sqrt(2)
        elif d == 3:
            print("TODO")
        psi[n - 1] = tn.Node(baseRightTensor, name=('site' + str(n - 1)),
                                   axis_names=['v' + str(n - 1), 's' + str(n - 1), 'v' + str(n)],
                                   backend=None)
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
                         backend=None)
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
                             backend=None)
            else:
                psi[i] = tn.Node(baseMiddleTensorOdd, name=('site' + str(i)),
                                 axis_names=['v' + str(i), 's' + str(i), 'v' + str(i + 1)],
                                 backend=None)
        if d == 2:
            baseRightTensor = np.zeros((2, 2, 1), dtype=complex)
            baseRightTensor[0, 1, 0] = 1
            baseRightTensor[1, 0, 0] = 1
        elif d == 3:
            baseRightTensor = np.zeros((1, 3, 1), dtype=complex)
            baseRightTensor[0, 2, 0] = 1
        psi[n - 1] = tn.Node(baseRightTensor, name=('site' + str(n - 1)),
                             axis_names=['v' + str(n - 1), 's' + str(n - 1), 'v' + str(n)],
                             backend=None)
        norm = getOverlap(psi, psi)
        psi[n - 1] = multNode(psi[n - 1], 1 / np.sqrt(norm))
        return psi
    elif mode == 'aklt':
        baseTensor = np.zeros((2, 3, 2), dtype=complex)
        baseTensor[0, 0, 1] = np.sqrt(2 / 3)
        baseTensor[0, 1, 0] = -np.sqrt(1 / 3)
        baseTensor[1, 1, 1] = np.sqrt(1 / 3)
        baseTensor[1, 2, 0] = -np.sqrt(2 / 3)
        for i in range(n):
            psi[i] = tn.Node(baseTensor, name=('site' + str(i)),
                                 axis_names=['v' + str(i), 's' + str(i), 'v' + str(i + 1)],
                                 backend=None)
        norm = getOverlap(psi, psi)
        psi[n - 1] = multNode(psi[n - 1], 1 / np.sqrt(norm))
        return psi
    elif mode == 'pbc':
        connectorsUnifierTensor = getLegsUnifierTensor(2, 2)
        physicalUnifierTensor = getLegsUnifierTensor(d, d)
        baseTensor = np.zeros((2, 3, 2), dtype=complex)
        baseTensor[0, 0, 1] = np.sqrt(2 / 3)
        baseTensor[0, 1, 0] = -np.sqrt(1 / 3)
        baseTensor[1, 1, 1] = np.sqrt(1 / 3)
        baseTensor[1, 2, 0] = -np.sqrt(2 / 3)
        baseLeftTensor = np.tensordot(baseTensor, baseTensor, axes=([0], [2]))
        baseLeftTensor = np.tensordot(baseLeftTensor, physicalUnifierTensor, axes=([0, 3], [0, 1]))
        baseLeftTensor = np.tensordot(baseLeftTensor, connectorsUnifierTensor, axes=([0, 1], [0, 1]))
        baseLeftTensor = np.reshape(baseLeftTensor, [1, baseLeftTensor.shape[0], baseLeftTensor.shape[1]])
        psi[0] = tn.Node(baseLeftTensor, name='site0', axis_names=['v0', 's0', 'v1'],
                         backend=None)
        baseMiddleTensor = np.tensordot(connectorsUnifierTensor, baseTensor, axes=([0], [0]))
        baseMiddleTensor = np.tensordot(baseMiddleTensor, baseTensor, axes=([0], [2]))
        baseMiddleTensor = np.tensordot(baseMiddleTensor, physicalUnifierTensor, axes=([1, 4], [0, 1]))
        baseMiddleTensor = np.tensordot(baseMiddleTensor, connectorsUnifierTensor, axes=([1, 2], [0, 1]))
        for i in range(1, n-1):
            psi[i] = tn.Node(baseMiddleTensor, name=('site' + str(i)),
                                   axis_names=['v' + str(i), 's' + str(i), 'v' + str(i+1)],
                                   backend=None)
        baseRightTensor = np.tensordot(baseTensor, baseTensor, axes=([2], [0]))
        baseRightTensor = np.tensordot(connectorsUnifierTensor, baseRightTensor, axes=([0, 1], [0, 3]))
        baseRightTensor = np.tensordot(baseRightTensor, physicalUnifierTensor, axes=([1, 2], [0, 1]))
        baseRightTensor = np.reshape(baseRightTensor, [baseRightTensor.shape[0], baseRightTensor.shape[1], 1])
        psi[n - 1] = tn.Node(baseRightTensor, name=('site' + str(n - 1)),
                                   axis_names=['v' + str(n - 1), 's' + str(n - 1), 'v' + str(n)],
                                   backend=None)
        norm = getOverlap(psi, psi)
        psi[n-1] = multNode(psi[n-1], 1/np.sqrt(norm))
        return psi

# Assuming psi1, psi2 have the same length, Hilbert space etc.
# assuming psi2 is conjugated
def getOverlap(psi1Orig: List[tn.Node], psi2Orig: List[tn.Node]):
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

def addNodes(node1, node2):
    # TODO asserts
    if node1 is None:
        if node2 is None:
            return None
        else:
            return node2
    else:
        if node2 is None:
            return node1
        else:
            result = copyState([node1])[0]
            result.set_tensor(result.get_tensor() + node2.get_tensor())
            return result

def multNode(node, c):
    node.set_tensor(node.get_tensor() * c)
    return node


def getNodeNorm(node):
    copy = copyState([node])[0]
    copyConj = copyState([node], conj=True)[0]
    for i in range(node.get_rank()):
        copy[i] ^ copyConj[i]
    return np.sqrt(tn.contract_between(copy, copyConj).get_tensor())


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
    node.tensor = np.transpose(node.tensor,
                               [edgeNum] + [i for i in range(len(node.edges)) if i != edgeNum])
    for i in range(node[0].dimension):
        node.tensor[i] *= diag[i]
    node.tensor = np.transpose(node.tensor,
                               list(range(1, edgeNum + 1)) + [0] + list(range(edgeNum + 1, len(node.edges))))
    return node


def permute(node: tn.Node, permutation) -> tn.Node:
    if node is None:
        return None
    axisNames = []
    for i in range(len(permutation)):
        axisNames.append(node.edges[permutation[i]].name)
    result = tn.transpose(node, permutation, axis_names=axisNames)
    if len(set(axisNames)) == len(axisNames):
        result.add_axis_names(axisNames)
    for i in range(len(axisNames)):
        result.get_edge(i).set_name(axisNames[i])
    result.set_name(node.name)
    tn.remove_node(node)
    return result


def svdTruncation(node: tn.Node, leftEdges: List[int], rightEdges: List[int],
                  dir: str, maxBondDim=128, leftName='U', rightName='V',  edgeName='default', normalize=False, maxTrunc=8):
    maxBondDim = getAppropriateMaxBondDim(maxBondDim,
                                          [node.edges[e] for e in leftEdges], [node.edges[e] for e in rightEdges])
    if dir == '>>':
        leftEdgeName = edgeName
        rightEdgeName = None
    else:
        leftEdgeName = None
        rightEdgeName = edgeName

    [U, S, V, truncErr] = tn.split_node_full_svd(node, [node.edges[e] for e in leftEdges],
                                                 [node.edges[e] for e in rightEdges], max_singular_values=maxBondDim,
                                       left_name=leftName, right_name=rightName,
                                       left_edge_name=leftEdgeName, right_edge_name=rightEdgeName)
    s = S
    S = tn.Node(np.diag(S.tensor))
    tn.remove_node(s)
    norm = np.sqrt(sum(S.tensor**2))
    if maxTrunc > 0:
        meaningful = sum(np.round(S.tensor / norm, maxTrunc) > 0)
        S.tensor = S.tensor[:meaningful]
        U.tensor = np.transpose(np.transpose(U.tensor)[:meaningful])
        V.tensor = V.tensor[:meaningful]
    if normalize:
        S = multNode(S, 1 / norm)
    for e in S.edges:
        e.name = edgeName
    if dir == '>>':
        l = copyState([U])[0]
        r = multiContraction(S, V, '1', '0', cleanOr1=True, cleanOr2=True, isDiag1=True)
    elif dir == '<<':
        l = multiContraction(U, S, [len(U.edges) - 1], '0', cleanOr1=True, cleanOr2=True, isDiag2=True)
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


def getRenyiEntropy(psi: List[tn.Node], n: int, AEnd: int, maxBondDim=256):
    psiCopy = copyState(psi)
    for k in [len(psiCopy) - 1 - i for i in range(len(psiCopy) - AEnd - 1)]:
        psiCopy = shiftWorkingSite(psiCopy, k, '<<')
    psiCopy[k - 1][2] ^ psiCopy[k][0]
    M = tn.contract_between(psi[k - 1], psi[k])

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
def assignNewSiteTensors(psi, k, M, dir, getOrthogonal=False):
    [sitek, sitekPlus1, truncErr] = svdTruncation(M, [M[0], M[1]], [M[2], M[3]], dir, \
            leftName=('site' + str(k)), rightName=('site' + str(k+1)), edgeName = ('v' + str(k+1)))
    tn.remove_node(psi[k])
    psi[k] = sitek
    # if k > 0:
    #     psi[k-1][2] ^ psi[k]
    tn.remove_node(psi[k+1])
    psi[k+1] = sitekPlus1
    # if k+2 < len(psi):
    #     psi[k+1][2] ^ psi[k+2][0]
    return [psi, truncErr]


def getEdgeNames(node: tn.Node):
    result = []
    for edge in node.edges:
        result.append(edge.name)
    return result


# k is curr working site, shift it by one in dir direction.
def shiftWorkingSite(psi: List[tn.Node], k, dir):
    if dir == '<<':
        pair = tn.contract(psi[k-1][2] ^ psi[k][0], axis_names=getEdgeNames(psi[k-1])[:2] + getEdgeNames(psi[k])[1:])
        [psi, truncErr] = assignNewSiteTensors(psi, k-1, pair, dir)
    else:
        pair = tn.contract(psi[k][2] ^ psi[k+1][0], axis_names=getEdgeNames(psi[k])[:2] + getEdgeNames(psi[k+1])[1:])
        [psi, truncErr] = assignNewSiteTensors(psi, k, pair, dir)
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
    psi[i] = permute(tn.contract_between(psi[i], op), [0, 2, 1])

