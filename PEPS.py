import numpy as np
import basicOperations as bops
import tensornetwork as tn
import gc

def bmpsRowStep(gammaL, lambdaMid, gammaR, lambdaSide, envOp, lattice='squared', chi=128, shrink=True):
    if lattice == 'squared':
        row = bops.multiContraction(bops.multiContraction(
            bops.multiContraction(bops.multiContraction(lambdaSide, gammaL, '1', '0', isDiag1=True),
                                  lambdaMid, '2', '0', cleanOr1=True, cleanOr2=True, isDiag2=True),
                                  gammaR, '2', '0', cleanOr1=True, cleanOr2=True),
                                  lambdaSide, '3', '0', cleanOr1=True, isDiag2=True)
        opRow = tn.Node(bops.multiContraction(row, envOp, '12', '01', cleanOr1=True).tensor\
                        .transpose([0, 2, 4, 5, 1, 3]))

        leftEdges = [0, 1, 2]
        rightEdges = [3, 4, 5]
        sideLegsNumber = 2
        siteIndexNumber = 3
    elif lattice == 'triangular':
        row = bops.multiContraction(bops.multiContraction(
            bops.multiContraction(bops.multiContraction(lambdaSide, gammaL, '1', '0', isDiag1=True),
                                  lambdaMid, '3', '0', cleanOr1=True, cleanOr2=True, isDiag2=True),
                                  gammaR, '3', '0', cleanOr1=True, cleanOr2=True),
                                  lambdaSide, '5', '0', cleanOr1=True, isDiag2=True)
        opRow = tn.Node(bops.multiContraction(row, envOp, '123', '156', cleanOr1=True).tensor.transpose(
                             [0, 3, 6, 5, 4, 9, 8, 2, 1, 7]))
        leftEdges = list(range(5))
        rightEdges = list(range(5, 10))
        sideLegsNumber = 3
        siteIndexNumber = 4
    else: # elif lattice == 'unionJack':
        row = bops.multiContraction(bops.multiContraction(
            bops.multiContraction(bops.multiContraction(lambdaSide, gammaL, '1', '0', isDiag1=True),
                                  lambdaMid, '4', '0', cleanOr1=True, cleanOr2=True, isDiag2=True),
                                  gammaR, '4', '0', cleanOr1=True, cleanOr2=True),
                                  lambdaSide, '7', '0', cleanOr1=True, isDiag2=True)
        # envOp =
        #    \ /
        #     O
        #  \|/ \|/
        # --O---O--
        #  /|\ /|\
        # Op order: top-left-right
        # in site ops, index order is top middle and anti-clockwise
        opRow = tn.Node(bops.multiContraction(row, envOp, '2345', '2018', cleanOr1=True).tensor.transpose(
            [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 3, 2, 13, 12]))
        leftEdges = list(range(7))
        rightEdges = list(range(7, 14))
        sideLegsNumber = 4
        siteIndexNumber = 5
    [U, S, V, truncErr] = bops.svdTruncation(opRow, leftEdges, rightEdges, dir='>*<', maxBondDim=chi, normalize=True)
    if len(truncErr) > 0:
        if np.max(truncErr) > 1e-5:
            # print(np.max(truncErr))
            b = 1
    newLambdaMid = tn.Node(np.diag(S.tensor) / np.sqrt(sum(np.diag(S.tensor)**2))) # bops.multNode(S, 1 / np.sqrt(sum(S.tensor**2)))
    lambdaSideInv = tn.Node(np.array([1 / val if val > 1e-15 else 0 for val in lambdaSide.tensor], dtype=gammaL.dtype))
    newGammaL = bops.multiContraction(lambdaSideInv, U, '1', '0', cleanOr2=True, isDiag1=True)
    splitter = tn.Node(bops.getLegsSplitterTensor([newGammaL[i].dimension for i in range(sideLegsNumber)]))
    newGammaR = bops.multiContraction(V, lambdaSideInv, [siteIndexNumber - 1], '0', cleanOr1=True, cleanOr2=True, isDiag2=True)
    newGammaL = bops.unifyLegs(newGammaL, list(range(sideLegsNumber)))
    newGammaR = bops.unifyLegs(newGammaR,
                        list(range(len(newGammaR.tensor.shape) - sideLegsNumber, len(newGammaR.tensor.shape))))
    newLambdaSide = bops.multiContraction(bops.multiContraction(
        lambdaSide, splitter, '1', '0', cleanOr1=True, isDiag1=True),
        splitter, list(range(sideLegsNumber)), list(range(sideLegsNumber)), cleanOr1=True, cleanOr2=True)
    temp = newLambdaSide
    newLambdaSide = tn.Node(np.diag(newLambdaSide.tensor))
    tn.remove_node(temp)
    if shrink:
        b = 1
    return newGammaL, newLambdaMid, newGammaR, newLambdaSide

def fidelity(rho, sigma):
    if np.all(np.round(rho, 13) == np.round(sigma, 13)):
        return 1
    vals, u = np.linalg.eigh(rho)
    vals = vals.astype('complex')
    vals = np.sqrt(vals)
    uSigmaU = np.matmul(np.conj(np.transpose(u)), np.matmul(sigma, u))
    sqrtRhoSigSqrtRho = np.matmul(np.diag(vals), np.matmul(uSigmaU, np.diag(vals)))
    vals = np.linalg.eigvalsh(sqrtRhoSigSqrtRho)
    # round to get rid of small negative numerical errors in vals
    vals = np.round(vals, 8)
    return sum(np.sqrt(vals))**2


def checkConvergence(oldGammaC, oldLambdaC, oldGammaD, oldLambdaD, GammaC, LambdaC, GammaD, LambdaD, d):
    dmC = np.round(getRowDM(GammaC, LambdaC, GammaD, LambdaD, 0, d), 16)
    oldDmC = np.round(getRowDM(oldGammaC, oldLambdaC, oldGammaD, oldLambdaD, 0, d), 16)
    dmD = np.round(getRowDM(GammaD, LambdaD, GammaC, LambdaC, 0, d), 16)
    oldDmD = np.round(getRowDM(oldGammaD, oldLambdaD, oldGammaC, oldLambdaC, 0, d), 16)
    return fidelity(dmC, oldDmC), fidelity(dmD, oldDmD)


def getRowDM(GammaL, LambdaL, GammaR, LambdaR, sites, d):
    c = bops.contract(bops.contract(LambdaR, GammaL, '1', '0', isDiag1=True), LambdaL, '2', '0', isDiag2=True)
    row = bops.contract(bops.contract(c, GammaR, '2', '0'), LambdaR, '3', '0', isDiag2=True)
    for i in range(sites):
        row = bops.multiContraction(row, GammaL, [len(row.edges) - 1], [0])
        row = bops.multiContraction(row, LambdaL, [len(row.edges) - 1], [0], isDiag2=True)
        row = bops.multiContraction(row, GammaR, [len(row.edges) - 1], [0])
        row = bops.multiContraction(row, LambdaR, [len(row.edges) - 1], [0], isDiag2=True)
    dm = bops.multiContraction(row, row, [0, len(row.edges) - 1], [0, len(row.edges) - 1, '*'])
    rho = np.reshape(dm.tensor, [d**((2 + 2 * sites)), d**((2 + 2 * sites))])
    return rho / np.trace(rho)


def getBMPSRowOps(GammaC, LambdaC, GammaD, LambdaD, AEnv, BEnv, steps, chi):
    convergence = []
    envOpAB = bops.permute(bops.multiContraction(AEnv, BEnv, '1', '3'), [0, 3, 2, 4, 1, 5])
    envOpBA = bops.permute(bops.multiContraction(BEnv, AEnv, '1', '3'), [0, 3, 2, 4, 1, 5])
    op = envOpBA
    for i in range(steps):
        oldGammaC, oldLambdaC, oldGammaD, oldLambdaD = GammaC, LambdaC, GammaD, LambdaD
        GammaC, LambdaC, GammaD, LambdaD = bmpsRowStep(GammaC, LambdaC, GammaD, LambdaD, op, chi=chi)
        GammaD, LambdaD, GammaC, LambdaC = bmpsRowStep(GammaD, LambdaD, GammaC, LambdaC, op, chi=chi)
        if i > 0:
            convergence.append(checkConvergence(oldGammaC, oldLambdaC, oldGammaD, oldLambdaD,
                                 GammaC, LambdaC, GammaD, LambdaD, AEnv[0].dimension))
        bops.removeState([oldGammaC, oldLambdaC, oldGammaD, oldLambdaD])
        gc.collect()
    bops.removeState([GammaC, LambdaC, GammaD, LambdaD, oldGammaC, oldLambdaC, oldGammaD, oldLambdaD])
    # print(np.round(convergence, 8))
    # plt.plot(convergence)
    # plt.show()
    return GammaC, LambdaC, GammaD, LambdaD


def bmpsSides(cUp: tn.Node, dUp: tn.Node, cDown: tn.Node, dDown: tn.Node, AEnv: tn.Node, BEnv: tn.Node, steps,
                 option='right'):
    envOpAB = bops.permute(bops.multiContraction(AEnv, BEnv, '1', '3'), [0, 3, 2, 4, 1, 5])
    upRow = bops.multiContraction(cUp, dUp, '2', '0')
    downRow = bops.multiContraction(cDown, dDown, '2', '0')
    if option == 'right':
        X = tn.Node(np.ones((upRow[3].dimension, envOpAB[3].dimension, downRow[3].dimension),
                            dtype=cUp.dtype))
    else:
        X = tn.Node(np.ones((upRow[0].dimension, envOpAB[2].dimension, downRow[0].dimension),
                            dtype=cUp.dtype))
    for i in range(steps):
        if option == 'right':
            X = bops.multiContraction(bops.multiContraction(bops.multiContraction(
                X, upRow, '0', '3'), envOpAB, '340', '013', cleanOr1=True), downRow, '034', '312')
        else:
            X = bops.multiContraction(bops.multiContraction(bops.multiContraction(
                X, upRow, '0', '0'), envOpAB, '023', '201', cleanOr1=True), downRow, '034', '012', cleanOr1=True)
        norm = np.sqrt(bops.multiContraction(X, X, '012', '012*').tensor)
        X = bops.multNode(X, 1 / norm)
    return X


#
#    O---0---0--
#   / \ / \ / \ /
#  O---A---B---A--
#   \ / \ / \ / \
#    O---0---0--
# 0 Is obtained in getBMPSRowOps. We are looking for O.
# We obtain
#    O--
#   / \
#  O---
#   \ /
#    O--
# And then decompose it to get
#    O--
#   / \
# The decomposition direction is determined by the attachment of lambda to gamma in 0:
#  edgeOp = --L--G--
#               / \
def bmpsTriangularCorner(edgeOp:tn.Node, AEnv: tn.Node, BEnv: tn.Node, steps):
    d = AEnv[0].dimension
    left = tn.Node(np.ones((edgeOp[3].dimension, d, d, d, edgeOp[0].dimension)))
    right = tn.Node(np.ones((edgeOp[3].dimension, d, d, d, edgeOp[0].dimension)))
    for i in range(steps):
        left = bops.multiContraction(bops.multiContraction(bops.multiContraction(left, edgeOp, '0', '3'),
                                                           AEnv, '0126', '4503'), edgeOp, '03', '01')
        left = bops.multiContraction(bops.multiContraction(bops.multiContraction(left, edgeOp, '0', '3'),
                                                           BEnv, '0126', '4503'), edgeOp, '03', '01')
        # right = bops.multiContraction(bops.multiContraction(bops.multiContraction(right, edgeOp, '0', '3'),
        #                                                     BEnv, '6012', '0123'), edgeOp, '03', '01')
        # right = bops.multiContraction(bops.multiContraction(bops.multiContraction(right, edgeOp, '0', '3'),
        #                                                     AEnv, '6012', '0123'), edgeOp, '03', '01')
    [d, leftX, te] = bops.svdTruncation(left, [0, 1, 2], [3, 4], '>>', normalize=True)
    [edgeOp, r, te] = bops.svdTruncation(edgeOp, [0, 1, 2], [3], '<<', maxBondDim=leftX[0].dimension, normalize=True)
    [d, rightX, te] = bops.svdTruncation(right, [0, 1, 2], [3, 4], '>>', normalize=True)
    return leftX, edgeOp



# Start with a 2*2 DM, increase later
def bmpsCols(upRow: tn.Node, downRow: tn.Node, AEnv: tn.Node, BEnv: tn.Node, steps,
                 option='right', X=None):
    envOpAB = bops.permute(bops.multiContraction(AEnv, BEnv, '1', '3'), [0, 3, 2, 4, 1, 5])
    envOpBA = bops.permute(bops.multiContraction(BEnv, AEnv, '1', '3'), [0, 3, 2, 4, 1, 5])
    if X is None:
        if option == 'right':
            X = tn.Node(np.ones((upRow[3].dimension, envOpBA[3].dimension, envOpAB[3].dimension, downRow[0].dimension),
                                dtype=upRow.dtype))
        else:
            X = tn.Node(np.ones((downRow[0].dimension, envOpAB[2].dimension, envOpBA[2].dimension, upRow[0].dimension),
                                dtype=upRow.dtype))
    for i in range(steps):
        if option == 'right':
            X = bops.multiContraction(upRow, X, '3', '0')
            X = bops.multiContraction(X, envOpBA, '123', '013', cleanOr1=True)
            X = bops.multiContraction(X, envOpAB, '451', '013', cleanOr1=True)
            X = bops.multiContraction(X, downRow, '154', '012', cleanOr1=True)
        else:
            X = bops.multiContraction(downRow, X, '3', '0')
            X = bops.multiContraction(X, envOpAB, '321', '245', cleanOr1=True)
            X = bops.multiContraction(X, envOpBA, '134', '245', cleanOr1=True)
            X = bops.multiContraction(X, upRow, '134', '012', cleanOr1=True)
        norm = np.sqrt(bops.multiContraction(X, X, '0123', '0123*').tensor)
        X = bops.multNode(X, 1 / norm)
    return X


def bmpsDensityMatrix(up_row, down_row, AEnv, BEnv, A, B, steps):
    rightRow = bmpsCols(up_row, down_row, AEnv, BEnv, steps, 'right')
    leftRow = bmpsCols(up_row, down_row, AEnv, BEnv, steps, 'left')
    circle = bops.multiContraction(
        bops.multiContraction(bops.multiContraction(leftRow, up_row, '3', '0'), rightRow, '5', '0'), down_row, '07', '03')


    parityTensor = np.eye(4, dtype=up_row.dtype)
    parityTensor[1, 1] = -1
    parityTensor[3, 3] = -1
    parity = tn.Node(parityTensor)
    if A[0].dimension == 4:
        parityA = tn.Node(np.trace(bops.multiContraction(parity, A, '1', '0').tensor, axis1=0, axis2=5))
        ABNet = bops.permute(
            bops.multiContraction(bops.multiContraction(parityA, parityA, '1', '3'), bops.multiContraction(parityA, parityA, '1', '3'), '15', '03',
                                  cleanOr1=True, cleanOr2=True),
            [5, 1, 0, 2, 3, 6, 4, 7])
        p2 = bops.multiContraction(circle, ABNet, '01234567', '01234567')
        b = 1

    ABNet = bops.permute(
        bops.multiContraction(bops.multiContraction(B, A, '2', '4'), bops.multiContraction(A, B, '2', '4'), '28', '16',
                              cleanOr1=True, cleanOr2=True),
        [2, 10, 9, 13, 14, 5, 1, 6, 0, 4, 8, 12, 3, 7, 11, 15])
    dm = bops.multiContraction(circle, ABNet, '23140567', '01234567', cleanOr1=True, cleanOr2=True)
    ordered = np.round(np.reshape(dm.tensor, [16,  16]), 13)
    ordered = ordered / np.trace(ordered)
    return dm


# https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.109.020505
# temp
def twoCopiesEntanglement(circle, A, B):
    doubleCircle = tn.Node(np.kron(circle.tensor, circle.tensor))
    doubleA = tn.Node(np.kron(A.tensor, A.tensor))
    doubleB = tn.Node(np.kron(B.tensor, B.tensor))

    AEnv = tn.Node(np.trace(A.get_tensor(), axis1=0, axis2=5))
    BEnv = tn.Node(np.trace(B.get_tensor(), axis1=0, axis2=5))
    ABNet = bops.permute(bops.multiContraction(bops.multiContraction(BEnv, AEnv, '1', '3'),
                                               bops.multiContraction(AEnv, BEnv, '1', '3'), '15', '03'),
                         [5, 1, 0, 2, 3, 6, 4, 7])
    n = bops.multiContraction(circle, ABNet, '01234567', '01234567').tensor

    p2 = bops.multiContraction(doubleCircle, ABNet, '01234567', '01234567').tensor
    return p2


# TODO I stopped getting A, B and return openA, openB, which would cause problems with old code.
#  Fix this by checking the case (for example, shape of A)
def applyBMPS(AEnv: tn.Node, BEnv:tn.Node, d=2, steps=100, chi=32):
    envOpAB = bops.permute(bops.multiContraction(AEnv, BEnv, '1', '3'), [0, 3, 2, 4, 1, 5])
    envOpBA = bops.permute(bops.multiContraction(BEnv, AEnv, '1', '3'), [0, 3, 2, 4, 1, 5])


    rowTensor = np.ones((11, AEnv[0].dimension, BEnv[0].dimension, 11), dtype=AEnv.dtype)
    row = tn.Node(rowTensor)
    #
    # upRow = bops.unifyLegs(bops.unifyLegs(bops.unifyLegs(bops.unifyLegs(
    #     bops.permute(bops.multiContraction(row, tn.Node(row.tensor), '12', '04'),
    #                  [0, 2, 3, 4, 7, 1, 5, 6]), [5, 6]),
    #     [5, 6]), [0, 1]), [0, 1])
    upRow = row
    [C, D, te] = bops.svdTruncation(upRow, [0, 1], [2, 3], '>>', normalize=True)
    upRow = bops.multiContraction(D, C, '2', '0')
    [cUp_orig, dUp_orig, te] = bops.svdTruncation(upRow, [0, 1], [2, 3], '>>', normalize=True)
    print('starting bmps rows')
    GammaC, LambdaC, GammaD, LambdaD = getBMPSRowOps(cUp_orig, tn.Node(np.ones(cUp_orig[2].dimension)), dUp_orig,
                                                          tn.Node(np.ones(dUp_orig[2].dimension)), AEnv, BEnv, steps, chi=chi)
    print('finished up')
    cUp = bops.multiContraction(GammaC, LambdaC, '2', '0', isDiag2=True)
    dUp = bops.multiContraction(GammaD, LambdaD, '2', '0', isDiag2=True)
    upRow = bops.multiContraction(cUp, dUp, '2', '0')

    GammaC, LambdaC, GammaD, LambdaD = getBMPSRowOps(cUp, tn.Node(np.ones(cUp[2].dimension)), dUp,
                                                          tn.Node(np.ones(dUp[2].dimension)),
                                                     bops.permute(AEnv, [2, 3, 0, 1]),
                                                     bops.permute(BEnv, [2, 3, 0, 1]), steps, chi=upRow.tensor.shape[0])
    print('finished down')
    cDown = bops.multiContraction(GammaC, LambdaC, '2', '0', isDiag2=True)
    dDown = bops.multiContraction(GammaD, LambdaD, '2', '0', isDiag2=True)
    downRow = bops.multiContraction(cDown, dDown, '2', '0')

    rightRow = bmpsCols(upRow, downRow, AEnv, BEnv, steps, option='right', X=None)
    leftRow = bmpsCols(upRow, downRow, AEnv, BEnv, steps, option='left', X=None)

    return upRow, downRow, leftRow, rightRow
