import numpy as np
import basicOperations as bops
import tensornetwork as tn
import PEPS as peps
import randomUs as ru
import pickle
import pepsExpect as pe


d = 2

def expectedDensityMatrix(height, width=2):
    if width != 2:
        # TODO
        return
    rho = np.zeros((d**(height * width), d**(height * width)))
    for i in range(d**(height * width)):
        b = 1
        for j in range(d**(height * width)):
            xors = i ^ j
            counter = 0
            # Look for pairs of reversed sites and count them
            while xors > 0:
                if xors & 3 == 3:
                    counter += 1
                elif xors & 3 == 1 or xors & 3 == 2:
                    counter = -1
                    xors = 0
                xors = xors >> 2
            if counter % 2 == 0:
                rho[i, j] = 1
    rho = rho / np.trace(rho)
    return rho


# Toric code model matrices - figure 30 here https://arxiv.org/pdf/1306.2164.pdf
baseTensor = np.zeros((d, d, d, d), dtype=complex)
baseTensor[0, 0, 0, 0] = 1 / 2**0.25
baseTensor[1, 0, 0, 1] = 1 / 2**0.25
baseTensor[0, 1, 1, 1] = 1 / 2**0.25
baseTensor[1, 1, 1, 0] = 1 / 2**0.25
base = tn.Node(baseTensor)


def get2ByNExplicit(l: int):
    with open('results/toricBoundaries', 'rb') as f:
        [upRow, downRow, leftRow, rightRow, openA, openB, A, B] = pickle.load(f)
    left = bops.multiContraction(downRow, bops.multiContraction(leftRow, upRow, '3', '0'), '3', '0')
    for i in range(1, l):
        left = bops.multiContraction(downRow, bops.multiContraction(left, upRow, [3 + 4 * i], '0', cleanOr1=True), '3', '0')
    circle = bops.multiContraction(left, downRow, [3 + 4 * l, 0], '03')
    openA = tn.Node(np.transpose(np.reshape(np.kron(A.tensor, A.tensor), [d ** 2, d ** 2, d ** 2, d ** 2, d, d]),
                                 [4, 0, 1, 2, 3, 5]))
    openB = tn.Node(np.transpose(np.reshape(np.kron(B.tensor, B.tensor), [d ** 2, d ** 2, d ** 2, d ** 2, d, d]),
                                 [4, 0, 1, 2, 3, 5]))
    ABNet = bops.permute(
        bops.multiContraction(bops.multiContraction(openB, openA, '2', '4'),
                              bops.multiContraction(openA, openB, '2', '4'), '28', '16',
                              cleanOr1=True, cleanOr2=True),
        [1, 5, 6, 13, 14, 9, 10, 2, 0, 4, 8, 12, 3, 7, 11, 15])
    if l == 2:
        curr = bops.multiContraction(circle, ABNet, '234567', '456701')
        res = bops.permute(bops.multiContraction(curr, ABNet, '23450176', '01234567'), [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15])
        rdm = np.reshape(res.tensor, [2**8, 2**8])
        rdm = rdm / np.trace(rdm)
    return rdm

def getExplicit2by2(g=0):
    with open('results/toricBoundaries_g_' + str(g), 'rb') as f:
        [upRow, downRow, leftRow, rightRow, openA, openB] = pickle.load(f)
    circle = bops.multiContraction(bops.multiContraction(bops.multiContraction(upRow, rightRow, '3', '0'), downRow, '5', '0'), leftRow, '70', '03')
    ABNet = bops.multiContraction(bops.multiContraction(openB, openA, '2', '4'), bops.multiContraction(openA, openB, '2', '4'), '28', '16',
                                  cleanOr1=True, cleanOr2=True)
    ABNet.tensor = ABNet.tensor.transpose([1, 5, 6, 13, 14, 9, 10, 2, 0, 4, 8, 12, 3, 7, 11, 15])
    dm = bops.multiContraction(circle, ABNet, '01234567', '01234567')
    ordered = np.round(np.reshape(dm.tensor, [16, 16]), 14)
    ordered /= np.trace(ordered)
    return ordered


gs = [k * 0.1 for k in range(11)]
gs = np.round(gs, 2)
renyis = []
a = False
if a:
    for g in gs:
        ABTensor = bops.multiContraction(base, base, '3', '0').tensor[0]
        ABTensor[0, 0, 0, 0, 0] *= (1 + g)
        ABTensor[1, 1, 1, 1, 0] *= (1 + g)
        A = tn.Node(ABTensor)
        B = tn.Node(np.transpose(ABTensor, [1, 2, 3, 0, 4]))

        AEnv = pe.toEnvOperator(bops.multiContraction(A, A, '4', '4*'))
        BEnv = pe.toEnvOperator(bops.multiContraction(B, B, '4', '4*'))
        chi = 32
        nonPhysicalLegs = 1
        GammaTensor = np.ones((nonPhysicalLegs, d**2, nonPhysicalLegs), dtype=complex)
        GammaC = tn.Node(GammaTensor, name='GammaC', backend=None)
        LambdaC = tn.Node(np.eye(nonPhysicalLegs) / np.sqrt(nonPhysicalLegs), backend=None)
        GammaD = tn.Node(GammaTensor, name='GammaD', backend=None)
        LambdaD = tn.Node(np.eye(nonPhysicalLegs) / np.sqrt(nonPhysicalLegs), backend=None)

        steps = 50

        envOpAB = bops.permute(bops.multiContraction(AEnv, BEnv, '1', '3'), [0, 3, 2, 4, 1, 5])
        envOpBA = bops.permute(bops.multiContraction(BEnv, AEnv, '1', '3'), [0, 3, 2, 4, 1, 5])
        curr = bops.permute(bops.multiContraction(envOpBA, envOpAB, '45', '01'), [0, 2, 4, 6, 1, 3, 5, 7])

        for i in range(2):
            [C, D, te] = bops.svdTruncation(curr, [0, 1, 2, 3], [4, 5, 6, 7], '>>', normalize=True)
            curr = bops.permute(bops.multiContraction(D, C, '23', '12'), [1, 3, 0, 5, 2, 4])
            curr = bops.permute(bops.multiContraction(curr, envOpAB, '45', '01'), [0, 2, 4, 6, 1, 3, 5, 7])

        currAB = curr

        openA = tn.Node(np.transpose(np.reshape(np.kron(A.tensor, A.tensor), [d**2, d**2, d**2, d**2, d, d]), [4, 0, 1, 2, 3, 5]))
        openB = tn.Node(np.transpose(np.reshape(np.kron(B.tensor, B.tensor), [d**2, d**2, d**2, d**2, d, d]), [4, 0, 1, 2, 3, 5]))

        rowTensor = np.zeros((11, 4, 4, 11), dtype=complex)
        rowTensor[0, 0, 0, 0] = 1
        rowTensor[1, 0, 0, 2] = 1
        rowTensor[2, 0, 0, 3] = 1
        rowTensor[3, 0, 3, 4] = 1
        rowTensor[4, 3, 0, 1] = 1
        rowTensor[5, 0, 0, 6] = 1
        rowTensor[6, 0, 3, 7] = 1
        rowTensor[7, 3, 0, 8] = 1
        rowTensor[8, 0, 0, 5] = 1
        row = tn.Node(rowTensor)

        upRow = bops.unifyLegs(bops.unifyLegs(bops.unifyLegs(bops.unifyLegs(
            bops.permute(bops.multiContraction(row, tn.Node(currAB.tensor), '12', '04'), [0, 2, 3, 4, 7, 1, 5, 6]), 5, 6), 5, 6), 0, 1), 0, 1)
        [C, D, te] = bops.svdTruncation(upRow, [0, 1], [2, 3], '>>', normalize=True)
        upRow = bops.multiContraction(D, C, '2', '0')
        [cUp, dUp, te] = bops.svdTruncation(upRow, [0, 1], [2, 3], '>>', normalize=True)
        GammaC, LambdaC, GammaD, LambdaD = peps.getBMPSRowOps(cUp, tn.Node(np.ones(cUp[2].dimension)), dUp,
                                                    tn.Node(np.ones(dUp[2].dimension)), AEnv, BEnv, 50)
        cUp = bops.multiContraction(GammaC, LambdaC, '2', '0', isDiag2=True)
        dUp = bops.multiContraction(GammaD, LambdaD, '2', '0', isDiag2=True)
        upRow = bops.multiContraction(cUp, dUp, '2', '0')
        downRow = bops.copyState([upRow])[0]
        rightRow = peps.bmpsCols(upRow, downRow, AEnv, BEnv, steps, option='right', X=upRow)
        leftRow = peps.bmpsCols(upRow, downRow, AEnv, BEnv, steps, option='left', X=upRow)
        with open('results/toricBoundaries_g_' + str(g), 'wb') as f:
            pickle.dump([upRow, downRow, leftRow, rightRow, openA, openB, A, B], f)
        print(g)
        ordered = getExplicit2by2(g)
        p2 = np.trace(np.matmul(ordered, ordered))
        renyi2 = -np.log(p2)
        renyis.append(renyi2)

    import matplotlib.pyplot as plt
    plt.plot(gs, renyis)
    plt.show()


def applyOpTosite(site, op):
    return pe.toEnvOperator(bops.multiContraction(bops.multiContraction(site, op, '4', '1'), site, '4', '4*'))


def applyVecsToSite(site: tn.Node, vecUp: np.array, vecDown: np.array):
    up = bops.multiContraction(site, tn.Node(vecUp), '4', '0', cleanOr2=True)
    down = bops.multiContraction(site, tn.Node(vecDown), '4*', '0*', cleanOr2=True)
    dm = tn.Node(np.kron(up.tensor, down.tensor))
    return dm


def horizontalPair(leftSite, rightSite, cleanLeft=True, cleanRight=True):
    pair = bops.multiContraction(leftSite, rightSite, '1', '3', cleanOr1=cleanLeft, cleanOr2=cleanRight)
    pair = bops.multiContraction(pair, ru.getPairUnitary(d), '37', '01')
    [left, right, te] = bops.svdTruncation(pair, [0, 1, 2, 6], [3, 4, 5, 7], '>>', maxBondDim=16)
    return bops.permute(left, [0, 4, 1, 2, 3]), bops.permute(right, [1, 2, 3, 0, 4])


def verticalPair(topSite, bottomSite, cleanTop=True, cleanBottom=True):
    pair = bops.multiContraction(topSite, bottomSite, '2', '0', cleanOr1=cleanTop, cleanOr2=cleanBottom)
    pair = bops.multiContraction(pair, ru.getPairUnitary(d), '37', '01',
                                 cleanOr1=True, cleanOr2=True)
    [top, bottom, te] = bops.svdTruncation(pair, [0, 1, 2, 6], [3, 4, 5, 7], '>>', maxBondDim=16)
    return bops.permute(top, [0, 1, 4, 2, 3]), bottom


def getPurity(w, h):
    with open('results/toricBoundaries', 'rb') as f:
        [upRow, downRow, leftRow, rightRow, openA, openB, A, B] = pickle.load(f)

    upRow = tn.Node(upRow)
    downRow = tn.Node(downRow)
    leftRow = tn.Node(leftRow)
    rightRow = tn.Node(rightRow)
    openA = tn.Node(openA)
    openB = tn.Node(openB)
    [cUp, dUp, te] = bops.svdTruncation(upRow, [0, 1], [2, 3], '>>')
    [cDown, dDown, te] = bops.svdTruncation(downRow, [0, 1], [2, 3], '>>')

    norm = pe.applyLocalOperators(cUp, dUp, cDown, dDown, leftRow, rightRow, A, B, w, h,
                               [tn.Node(np.eye(d)) for i in range(w * h)])
    leftRow = bops.multNode(leftRow, 1 / norm)
    res = pe.applyLocalOperators(cUp, dUp, cDown, dDown, leftRow, rightRow, A, B, w, h,
                            [tn.Node(ru.proj0Tensor) for i in range(w * h * 4)])
    # The density matrix is constructed of blocks of ones of size N and normalized by res.
    # Squaring it adds a factor of N * res.
    N = 2**(int(w * h / 4))
    purity = N * res
    return purity


def getToricGPurity(w, h, g):
    with open('results/toricBoundaries_g_' + str(g), 'rb') as f:
        [upRow, downRow, leftRow, rightRow, openA, openB] = pickle.load(f)
    D = openA.tensor.shape[1]
    A2 = tn.Node(np.reshape(np.transpose(bops.multiContraction(openA, openA, '50', '05').tensor,
                                         [0, 4, 1, 5, 2, 6, 3, 7]), [D**2, D**2, D**2, D**2]))
    B2 = tn.Node(np.reshape(np.transpose(bops.multiContraction(openB, openB, '50', '05').tensor,
                                         [0, 4, 1, 5, 2, 6, 3, 7]), [D**2, D**2, D**2, D**2]))
    upRow2 = tn.Node(np.kron(upRow.tensor, upRow.tensor))
    downRow2 = tn.Node(np.kron(downRow.tensor, downRow.tensor))
    leftRow2 = tn.Node(np.kron(leftRow.tensor, leftRow.tensor))
    rightRow2 = tn.Node(np.kron(rightRow.tensor, rightRow.tensor))
    if w == 2:
        left = leftRow2
        for i in range(int(h/2)):
            boundary = bops.multiContraction(bops.multiContraction(downRow2, left, '3', '0'), upRow2, '5', '0')
            left = tn.Node(bops.multiContraction(bops.multiContraction(bops.multiContraction(bops.multiContraction(
                boundary, B2, '45', '30'), A2, '723', '023'), A2, '42', '30'), B2, '513', '023').tensor. \
                           transpose([0, 3, 2, 1]))
        return bops.multiContraction(left, rightRow2, '0123', '3210').tensor * 1

# gs = np.round(np.array([0.1 * i for i in range(11)]), 1)
# for g in gs:
#     print([getToricGPurity(2, 2, g), np.trace(np.matmul(getExplicit2by2(g), getExplicit2by2(g)))])
