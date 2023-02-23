import numpy as np
import basicOperations as bops
import tensornetwork as tn
import randomUs as ru
import gc

d = 2


def toEnvOperator(op):
    op.reorder_axes([0, 4, 1, 5, 2, 6, 3, 7])
    result = bops.unifyLegs(bops.unifyLegs(bops.unifyLegs(bops.unifyLegs(
        op, [6, 7]), [4, 5]), [2, 3]), [0, 1])
    tn.remove_node(op)
    return result


def applyOpTosite(site, op):
    if len(site.tensor.shape) == 6: # site is in DM form
        if len(op.tensor.shape) == 2: # local op
            return bops.contract(site, op, '05', '10')
        else:
            return bops.unifyLegs(bops.unifyLegs(bops.unifyLegs(bops.permute(bops.contract(
                site, op, '05', '05'), [0, 4, 1, 5, 2, 6, 3, 7]), [6, 7]), [4, 5]), [2, 3], [0, 1])
    else:
        if len(op.tensor.shape) == 2:
            return toEnvOperator(bops.multiContraction(bops.multiContraction(site, op, '4', '1'), site, '4', '4*'))
        else:
            return bops.unifyLegs(bops.unifyLegs(bops.unifyLegs(bops.unifyLegs(bops.permute(
                bops.contract(bops.contract(site, op, '4', '0'), site, '8', '4*')),
                [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]),
                [9, 10, 11]), [6, 7, 8]), [3, 4, 5], [0, 1, 2])


def applyLocalOperators(cUp, dUp, cDown, dDown, leftRow, rightRow, A, B, h, w, ops, PBC=False, period_num=None):
    cUps = [cUp for i in range(int(w/2))]
    dUps = [dUp for i in range(int(w/2))]
    cDowns = [cDown for i in range(int(w/2))]
    dDowns = [dDown for i in range(int(w/2))]
    leftRows = [leftRow for i in range(int(h/2))]
    rightRows = [rightRow for i in range(int(h/2))]
    return applyLocalOperators_detailedBoundary(
        cUps, dUps, cDowns, dDowns, leftRows, rightRows, A, B, h, w, ops, PBC=PBC, period_num=period_num)


def applyLocalOperators_detailedBoundary(
        cUps, dUps, cDowns, dDowns, leftRows, rightRows, A, B, h, w, ops, PBC=False, period_num=None, envOps=None):
    if h == 2:
        if PBC:
            dims = [cUps[0][0].dimension, A[4].dimension, B[4].dimension, dDowns[0][2].dimension]
            left = tn.Node(np.eye(np.prod(dims)).reshape(dims + dims))
        else:
            left = leftRows[0]
        for wi in range(int(w / 2)):
            gc.collect()
            if envOps is None:
                leftUp = applyOpTosite(B, ops[wi * 4])
                leftDown = applyOpTosite(A, ops[wi * 4 + 1])
                rightUp = applyOpTosite(A, ops[wi * 4 + 2])
                rightDown = applyOpTosite(B, ops[wi * 4 + 3])
            else:
                leftUp, leftDown, rightUp, rightDown = envOps[wi * 4: wi * 4 + 4]
            if PBC:
                left = bops.contract(bops.contract(bops.contract(bops.contract(
                    left, cUps[wi], '4', '0'), leftUp, '47', '30'), leftDown, '48', '30'), dDowns[wi], '48', '21')
                left = bops.contract(bops.contract(bops.contract(bops.contract(
                    left, dUps[wi], '4', '0'), rightUp, '47', '30'), rightDown, '48', '30'), cDowns[wi], '48', '21')
            else:
                left = bops.multiContraction(left, cUps[wi], '3', '0', cleanOr1=True)
                left = bops.multiContraction(left, leftUp, '23', '30', cleanOr1=True)
                left = bops.multiContraction(left, leftDown, '14', '30', cleanOr1=True)
                left = bops.multiContraction(left, dDowns[wi], '04', '21', cleanOr1=True).reorder_axes([3, 2, 1, 0])

                left = bops.multiContraction(left, dUps[wi], '3', '0', cleanOr1=True)
                left = bops.multiContraction(left, rightUp, '23', '30', cleanOr1=True)
                left = bops.multiContraction(left, rightDown, '14', '30', cleanOr1=True)
                left = bops.multiContraction(left, cDowns[wi], '04', '21', cleanOr1=True).reorder_axes([3, 2, 1, 0])

        if PBC:
            return np.linalg.matrix_power(left.tensor.reshape([np.prod(dims)] * 2), period_num).trace()
        else:
            return bops.multiContraction(left, rightRows[0], '0123', '3210').tensor * 1
    elif h == 4:
        left = bops.multiContraction(leftRows[0], leftRows[1], '3', '0')
        for wi in range(int(w / 2)):
            gc.collect()
            left = bops.multiContraction(left, cUps[wi], '5', '0', cleanOr1=True)
            if envOps is None:
                left0 = applyOpTosite(B, ops[wi * 8])
                left1 = applyOpTosite(A, ops[wi * 8 + 1])
                left2 = applyOpTosite(B, ops[wi * 8 + 2])
                left3 = applyOpTosite(A, ops[wi * 8 + 3])
            else:
                left0, left1, left2, left3 = envOps[wi * 8: wi * 8 + 4]
            left = bops.multiContraction(left, left0, '45', '30', cleanOr1=True, cleanOr2=True)
            left = bops.multiContraction(left, left1, '36', '30', cleanOr1=True, cleanOr2=True)
            left = bops.multiContraction(left, left2, '26', '30', cleanOr1=True, cleanOr2=True)
            left = bops.multiContraction(left, left3, '16', '30', cleanOr1=True, cleanOr2=True)
            left = bops.multiContraction(left, dDowns[wi], '06', '21', cleanOr1=True).reorder_axes([5, 4, 3, 2, 1, 0])

            left = bops.multiContraction(left, dUps[wi], '5', '0', cleanOr1=True)
            if envOps is None:
                right0 = applyOpTosite(A, ops[wi * 8 + 4])
                right1 = applyOpTosite(B, ops[wi * 8 + 5])
                right2 = applyOpTosite(A, ops[wi * 8 + 6])
                right3 = applyOpTosite(B, ops[wi * 8 + 7])
            else:
                right0, right1, right2, right3 = envOps[wi * 8 + 4: wi * 8 + 8]
            left = bops.multiContraction(left, right0, '45', '30', cleanOr1=True, cleanOr2=True)
            left = bops.multiContraction(left, right1, '36', '30', cleanOr1=True, cleanOr2=True)
            left = bops.multiContraction(left, right2, '26', '30', cleanOr1=True, cleanOr2=True)
            left = bops.multiContraction(left, right3, '16', '30', cleanOr1=True, cleanOr2=True)
            left = bops.multiContraction(left, cDowns[wi], '06', '21', cleanOr1=True).reorder_axes([5, 4, 3, 2, 1, 0])
        right = bops.multiContraction(rightRows[0], rightRows[1], '3', '0')
        res = bops.multiContraction(left, right, '012345', '543210').tensor * 1
        return res
    else:
        left = leftRows[0]
        for i in range(1, int(h / 2)):
            left = bops.contract(left, leftRows[i], [2 * i + 1], '0')
        for wi in range(int(w / 2)):
            gc.collect()
            left = bops.multiContraction(left, cUps[wi], [h + 1], '0', cleanOr1=True)
            for hi in range(int(h/2)):
                left_B = applyOpTosite(B, ops[wi * h * 2 + hi * 2])
                left_A = applyOpTosite(A, ops[wi * h * 2 + hi * 2 + 1])
                if hi == 0:
                    left = bops.contract(left, left_B, [h, h + 1], '30', cleanOr1=True, cleanOr2=True)
                else:
                    left = bops.contract(left, left_B, [h - hi * 2, h + 2], '30', cleanOr1=True, cleanOr2=True)
                left = bops.contract(left, left_A, [h - hi * 2 - 1, h + 2], '30', cleanOr1=True, cleanOr2=True)
            left = bops.contract(left, dDowns[wi], [0, h + 2], '21', cleanOr1=True).reorder_axes(list(range(h + 1, -1, -1)))

            left = bops.multiContraction(left, dUps[wi], [h + 1], '0', cleanOr1=True)
            for hi in range(int(h/2)):
                left_A = applyOpTosite(A, ops[wi * h * 2 + h + hi * 2])
                left_B = applyOpTosite(B, ops[wi * h * 2 + h + hi * 2 + 1])
                if hi == 0:
                    left = bops.contract(left, left_A, [h, h + 1], '30', cleanOr1=True, cleanOr2=True)
                else:
                    left = bops.contract(left, left_A, [h - hi * 2, h + 2], '30', cleanOr1=True, cleanOr2=True)
                left = bops.contract(left, left_B, [h - hi * 2 - 1, h + 2], '30', cleanOr1=True, cleanOr2=True)
            left = bops.contract(left, cDowns[wi], [0, h + 2], '21', cleanOr1=True).reorder_axes(list(range(h + 1, -1, -1)))
        right = rightRows[0]
        for i in range(1, int(h / 2)):
            right = bops.contract(right, rightRows[i], [2 * i + 1], '0')
        res = bops.contract(left, right, list(range(h+2)), list(range(h+1, -1, -1))).tensor * 1
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


def applyLocalOperators_torus(cUp, dUp, cDown, dDown, leftRow, rightRow, A, B, w, h, ops):
    if w == 4:
        for i in range(int(h/2)):
            left0 = applyOpTosite(B, ops[i * 8])
            left1 = applyOpTosite(A, ops[i * 8 + 1])
            left2 = applyOpTosite(B, ops[i * 8 + 2])
            left3 = applyOpTosite(A, ops[i * 8 + 3])
            left = bops.permute(bops.multiContraction(left0, bops.multiContraction(left1,
                                        bops.multiContraction(left2, left3, '2', '0'), '2', '0'), '20', '06'),
                                [0, 2, 4, 6, 1, 3, 5, 7])

            right0 = applyOpTosite(A, ops[i * 8 + 4])
            right1 = applyOpTosite(B, ops[i * 8 + 5])
            right2 = applyOpTosite(A, ops[i * 8 + 6])
            right3 = applyOpTosite(B, ops[i * 8 + 7])
            right = bops.permute(bops.multiContraction(right0, bops.multiContraction(right1,
                                         bops.multiContraction(right2, right3, '2', '0'), '2', '0'), '20', '06'),
                            [0, 2, 4, 6, 1, 3, 5, 7])
            curr = bops.multiContraction(left, right, '4567', '0123')
            if i == 0:
                circle = curr
            elif i > 0 and i < h/2 - 1:
                circle = bops.multiContraction(circle, curr, '4567', '0123')
            else:
                result = bops.multiContraction(circle, curr, '45670123', '01234567').tensor
        return result
