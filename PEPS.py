from scipy import linalg
import numpy as np
import basicOperations as bops
import randomMeasurements as rm
import sys
import tensornetwork as tn

# Toric code model matrices
QTensor = np.zeros((2, 2, 2))
QTensor[0, 0, 0] = 1
QTensor[1, 1, 0] = 1
QTensor[1, 0, 1] = 1
QTensor[0, 1, 1] = 1
Q = tn.Node(QTensor, name='Q', backend=None)
deltaTensor = np.zeros((2, 2, 2))
deltaTensor[0, 0, 0] = 1
deltaTensor[1, 1, 1] = 1
delta = tn.Node(deltaTensor, name='delta', backend=None)

def getA():
    left = bops.multiContraction(Q, delta, '1', '0')
    A = bops.permute(bops.multiContraction(left, Q, '2', '1'), [0, 1, 4, 3, 2])
    tn.remove_node(left)
    return A


def getB():
    A = getA()
    B = bops.permute(A, [1, 2, 3, 0, 4])
    return B


A = getA()
B = getB()

for i in range(2):
    for j in range(2):
        for k in range(2):
            for l in range(2):
                for m in range(2):
                    if A.tensor[j ,k, l, m, i] != B.tensor[k, l, m, j, i]:
                        b = 1

AEnv = bops.permute(bops.multiContraction(A, A, '4', '4*'), [0, 4, 1, 5, 2, 6, 3, 7])
BEnv = bops.permute(bops.multiContraction(B, B, '4', '4*'), [0, 4, 1, 5, 2, 6, 3, 7])
chi = 32
# Double 'physical' leg for the closed MPS
CTensor = np.zeros((2, 2, 2, 2))
CTensor[0, 1, 0, 1] = -1
CTensor[1, 0, 1, 0] = -1
CTensor[0, 0, 0, 1] = 1
CTensor[1, 1, 1, 0] = 1
C = tn.Node(CTensor / np.sqrt(2), name='C', backend=None)
D = tn.Node(CTensor / np.sqrt(2), name='D', backend=None)

def bMPSRowStepRight(leftNode, rightNode, envNode, dirStr):
    pair = bops.multiContraction(leftNode, rightNode, '3', '0')
    layeredPair = bops.permute(bops.multiContraction(pair, envNode, '1234', '0123'), [0, 2, 3, 4, 5, 1])
    tn.remove_node(leftNode)
    tn.remove_node(rightNode)
    tn.remove_node(pair)
    [leftNode, rightNode, truncErr] = bops.svdTruncation(layeredPair, leftEdges=[layeredPair.edges[0], layeredPair.edges[1], layeredPair.edges[2]],
                                          rightEdges=[layeredPair.edges[3], layeredPair.edges[4], layeredPair.edges[5]],
                      dir=dirStr, maxBondDim=chi)
    tn.remove_node(layeredPair)
    return leftNode, rightNode

# def bMPSRowStepLeft(leftNode, rightNode, envNode, dirStr):
#     # TODO

def bMPSColStepB(X, C, D):
    bx = bops.multiContraction(X, BEnv, '12', '01')
    bxd = bops.multiContraction(bx, D, '123', '012*')
    bxdc = bops.multiContraction(bxd, C, '534', '012*')
    dbxdc = bops.multiContraction(D, bxdc, '0', '0')
    adbxdc = bops.multiContraction(dbxdc, AEnv,  '0134', '0123')
    final = bops.permute(bops.multiContraction(adbxdc, C, '023', '012'), [3, 1, 2, 0])
    # tn.remove_node(X)
    tn.remove_node(bx)
    tn.remove_node(bxdc)
    tn.remove_node(dbxdc)
    tn.remove_node(adbxdc)
    return final


def bMPSColStepA(X, C, D):
    ax = bops.multiContraction(X, AEnv, '12', '01')
    axd = bops.multiContraction(ax, D, '123', '012*')
    axdc = bops.multiContraction(axd, C, '534', '012*')
    daxdc = bops.multiContraction(D, axdc, '0', '0')
    bdaxdc = bops.multiContraction(daxdc, BEnv,  '0134', '0123')
    final = bops.permute(bops.multiContraction(bdaxdc, C, '023', '012'), [3, 1, 2, 0])
    # tn.remove_node(X)
    tn.remove_node(ax)
    tn.remove_node(axdc)
    tn.remove_node(daxdc)
    tn.remove_node(bdaxdc)
    return final


def convergenceTest(C, D):
    cdPair = bops.multiContraction(C, D, '3', '0')
    acd = bops.permute(bops.multiContraction(cdPair, AEnv, '1234', '0123'), [0, 2, 3, 4, 5, 1])
    norm = np.sqrt(bops.multiContraction(acd, acd, '012345', '012345*').tensor)
    acd = bops.multNode(acd, 1 / norm)
    aTest = bops.multiContraction(cdPair, acd, '012345', '012345*').tensor

    dcPair = bops.multiContraction(D, C, '3', '0')
    bdc = bops.permute(bops.multiContraction(dcPair, BEnv, '1234', '0123'), [0, 2, 3, 4, 5, 1])
    norm = np.sqrt(bops.multiContraction(bdc, bdc, '012345', '012345*').tensor)
    bdc = bops.multNode(bdc, 1 / norm)
    bTest = bops.multiContraction(dcPair, bdc, '012345', '012345*').tensor
    return aTest, bTest


pair = bops.multiContraction(C, D, '3', '0')
norm = np.sqrt(bops.multiContraction(pair, pair, '012345', '012345*').tensor)
D = bops.multNode(D, 1 / norm)

origPair = pair

aTests = []
bTests = []
aabTests = []

for i in range(199):
    C, D = bMPSRowStepRight(C, D, AEnv, '>>')
    id = bops.multiContraction(C, C, '012', '012*')
    C = bops.multNode(C, 1 / np.sqrt(id.tensor[0, 0]))
    D, C = bMPSRowStepRight(D, C, BEnv, '>>')
    id = bops.multiContraction(D, D, '123', '123*')
    D = bops.multNode(D, 1 / np.sqrt(id.tensor[0, 0]))
    pair = bops.multiContraction(C, D, '3', '0')
    norm = np.sqrt(bops.multiContraction(pair, pair, '012345', '012345*').tensor)
    C = bops.multNode(C, 1 / norm)
    aTest, bTest = convergenceTest(C, D) * 1

    aTests.append(aTest)
    bTests.append(bTest)

Xb = bops.multiContraction(pair, C, '5', '0')
Xb = bops.permute(bops.multiContraction(Xb, AEnv, '123456', '012345'), [0, 2, 3, 1])
xbtests = []
Xa = bops.multiContraction(D, pair, '3', '0')
Xa = bops.permute(bops.multiContraction(Xa, AEnv, '123456', '012345'), [0, 2, 3, 1])
xatests = []
for i in range(200):
    XbPrev = Xb
    Xb = bMPSColStepB(Xb, C, D)
    norm = np.sqrt(bops.multiContraction(Xb, Xb, '0123', '0123*').tensor)
    Xb = bops.multNode(Xb, 1 / norm)
    xbtests.append(np.sqrt(bops.multiContraction(Xb, XbPrev, '0123', '0123*').tensor))
    tn.remove_node(XbPrev)

    XaPrev = Xa
    Xa = bMPSColStepB(Xa, C, D)
    norm = np.sqrt(bops.multiContraction(Xa, Xa, '0123', '0123*').tensor)
    Xa = bops.multNode(Xa, 1 / norm)
    xatests.append(np.sqrt(bops.multiContraction(Xa, XaPrev, '0123', '0123*').tensor))
    tn.remove_node(XaPrev)

XbD = bops.multiContraction(Xb, D, '3', '0*')
XbDC = bops.multiContraction(XbD, C, '5', '0*')
XbDCB = bops.multiContraction(XbDC, B, '135', '012')
XbDCBB = bops.multiContraction(XbDCB, B, '123', '012*')

XaC = bops.multiContraction(Xa, C, '3*', '0')
XaCD = bops.multiContraction(XaC, D, '5', '0')
XaCDA = bops.multiContraction(XaCD, A, '135', '012')
XaCDAA = bops.multiContraction(XaCDA, A, '123', '012*')

DM = bops.permute(bops.multiContraction(XbDCBB, XaCDAA, '0124', '1024'), [0, 2, 1, 3])
dm = np.reshape(DM.tensor, [4, 4])
b = 1