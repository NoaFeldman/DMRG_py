import numpy as np
import basicOperations as bops
import tensornetwork as tn
import PEPS as peps
import randomUs as ru
import pickle

d = 2


def toEnvOperator(op):
    op.reorder_axes([0, 4, 1, 5, 2, 6, 3, 7])
    result = bops.unifyLegs(bops.unifyLegs(bops.unifyLegs(bops.unifyLegs(
        op, 6, 7), 4, 5), 2, 3), 0, 1)
    tn.remove_node(op)
    return result


def applyOpTosite(site, op):
    return toEnvOperator(bops.multiContraction(bops.multiContraction(site, op, '4', '1'), site, '4', '4*'))


def applyLocalOperators(cUp, dUp, cDown, dDown, leftRow, rightRow, A, B, w, h, ops):
    if w == 2:
        left = leftRow
        for i in range(int(h/2)):
            left = bops.multiContraction(left, cUp, '3', '0', cleanOr1=True)
            leftUp = applyOpTosite(B, ops[i * 4])
            leftDown = applyOpTosite(A, ops[i * 4 + 1])
            left = bops.multiContraction(left, leftUp, '23', '30', cleanOr1=True)
            left = bops.multiContraction(left, leftDown, '14', '30', cleanOr1=True)
            left = bops.multiContraction(left, dDown, '04', '21', cleanOr1=True).reorder_axes([3, 2, 1, 0])

            left = bops.multiContraction(left, dUp, '3', '0', cleanOr1=True)
            rightUp = applyOpTosite(A, ops[i * 4 + 2])
            rightDown = applyOpTosite(B, ops[i * 4 + 3])
            left = bops.multiContraction(left, rightUp, '23', '30', cleanOr1=True)
            left = bops.multiContraction(left, rightDown, '14', '30', cleanOr1=True)
            left = bops.multiContraction(left, cDown, '04', '21', cleanOr1=True).reorder_axes([3, 2, 1, 0])

            bops.removeState([leftUp, leftDown, rightDown, rightUp])

        return bops.multiContraction(left, rightRow, '0123', '3210').tensor * 1
    elif w == 4:
        left = bops.multiContraction(leftRow, leftRow, '3', '0')
        for i in range(int(h/2)):
            left = bops.multiContraction(left, cUp, '5', '0', cleanOr1=True)
            left0 = applyOpTosite(B, ops[i * 8])
            left1 = applyOpTosite(A, ops[i * 8 + 1])
            left2 = applyOpTosite(B, ops[i * 8 + 2])
            left3 = applyOpTosite(A, ops[i * 8 + 3])
            left = bops.multiContraction(left, left0, '45', '30', cleanOr1=True, cleanOr2=True)
            left = bops.multiContraction(left, left1, '36', '30', cleanOr1=True, cleanOr2=True)
            left = bops.multiContraction(left, left2, '26', '30', cleanOr1=True, cleanOr2=True)
            left = bops.multiContraction(left, left3, '16', '30', cleanOr1=True, cleanOr2=True)
            left = bops.multiContraction(left, dDown, '06', '21', cleanOr1=True).reorder_axes([5, 4, 3, 2, 1, 0])

            left = bops.multiContraction(left, dUp, '5', '0', cleanOr1=True)
            right0 = applyOpTosite(A, ops[i * 8 + 4])
            right1 = applyOpTosite(B, ops[i * 8 + 5])
            right2 = applyOpTosite(A, ops[i * 8 + 6])
            right3 = applyOpTosite(B, ops[i * 8 + 7])
            left = bops.multiContraction(left, right0, '45', '30', cleanOr1=True, cleanOr2=True)
            left = bops.multiContraction(left, right1, '36', '30', cleanOr1=True, cleanOr2=True)
            left = bops.multiContraction(left, right2, '26', '30', cleanOr1=True, cleanOr2=True)
            left = bops.multiContraction(left, right3, '16', '30', cleanOr1=True, cleanOr2=True)
            left = bops.multiContraction(left, cDown, '06', '21', cleanOr1=True).reorder_axes([5, 4, 3, 2, 1, 0])
        right = bops.multiContraction(rightRow, rightRow, '3', '0')
        res = bops.multiContraction(left, right, '012345', '543210').tensor * 1
        return res


def applyVecsToSite(site: tn.Node, vecUp: np.array, vecDown: np.array):
    up = bops.multiContraction(site, tn.Node(vecUp), '4', '0', cleanOr2=True)
    down = bops.multiContraction(site, tn.Node(vecDown), '4*', '0*', cleanOr2=True)
    dm = tn.Node(np.kron(up.tensor, down.tensor))
    return dm


def horizontalPair(leftSite, rightSite, cleanLeft=True, cleanRight=True):
    pair = bops.multiContraction(leftSite, rightSite, '1', '3', cleanOr1=cleanLeft, cleanOr2=cleanRight)
    pair = bops.multiContraction(pair, ru.getPairUnitary(d), '37', '01')
    [left, right, te] = bops.svdTruncation(pair, [0, 1, 2, 6], [3, 4, 5, 7], '>>', maxBondDim=16)
    left.reorder_axes([0, 4, 1, 2, 3])
    right.reorder.axis([1, 2, 3, 0, 4])
    return left, right


def verticalPair(topSite, bottomSite, cleanTop=True, cleanBottom=True):
    pair = bops.multiContraction(topSite, bottomSite, '2', '0', cleanOr1=cleanTop, cleanOr2=cleanBottom)
    pair = bops.multiContraction(pair, ru.getPairUnitary(d), '37', '01',
                                 cleanOr1=True, cleanOr2=True)
    [top, bottom, te] = bops.svdTruncation(pair, [0, 1, 2, 6], [3, 4, 5, 7], '>>', maxBondDim=16)
    top.reorder.axes([0, 1, 4, 2, 3])
    return top, bottom