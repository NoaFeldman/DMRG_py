import numpy as np
import basicOperations as bops
import tensornetwork as tn
import PEPS as peps
import scipy
import math

d = 2

"""A Random matrix distributed with Haar measure"""
def haar_measure(n):
    z = (np.random.randn(n, n) + 1j * np.random.randn(n, n)) / np.sqrt(2.0)
    q,r = scipy.linalg.qr(z)
    d = np.diagonal(r)
    ph = d / np.absolute(d)
    q = np.multiply(q,ph,q)
    return q

def randomMeas(rho):
    r = np.random.uniform()
    accum = 0
    for i in range(len(rho)):
        if r > accum and r < accum + rho[i, i]:
            return i
        accum += rho[i, i]
    b = 1


projectors = [tn.Node(np.zeros((d, d), dtype=complex)), tn.Node(np.zeros((d, d), dtype=complex))]
projectors[0].tensor[0, 0] = 1
projectors[1].tensor[1, 1] = 1

# Create top left corner and fill the rows. If I need another corner, handle it in the calling function by adjusting the
# projections.
def getClosedCorner(AEnv, BEnv, border, width, height, projections):
    if width == 0 or height == 0:
        return border
    for i in range(height):
        if projections[i, 0] == -1:
            startOp = BEnv
        else:
            startOp = projectors[projections[i, 0]]
        corner = bops.multiContraction(border, startOp, [width, width + 1], '03', cleanOr1=True)
        for j in range(1, width):
            if projections[i, j] == -1:
                if j % 2 == 1:
                    op = AEnv
                else:
                    op = BEnv
            else:
                op = projectors[projections[i, j]]
            corner = bops.multiContraction(corner, op, [width - j, width + height], '03', cleanOr1=True)
        corner = bops.permute(corner,
                              list(range(i + 1)) + [width + height] + list(range(i + 1, width + height)) + [width + height + 1])
    return corner

# Assuming even height and even width for the full system (gotten in getMeasurement),
# so the top row starts with a C and ends with a D and so on.
def getTopLeftCorner(AEnv, BEnv, C, D, width, height, projections):
    border = tn.Node(np.eye(C[2].dimension, dtype=complex))
    for i in range(height):
        if i % 2 == 0:
            op = D
        else:
            op = C
        border = bops.multiContraction(border, op, [len(border.edges) - 1], '0', cleanOr1=True)
    for i in range(width):
        if i % 0 == 0:
            op = C
        else:
            op = D
        border = bops.multiContraction(op, border, '2', '0', cleanOr2=True)
    if height == 0:
        border = bops.multiContraction(border, D, [len(border.edges) - 1], '0', cleanOr1=True)
        return border
    if width == 0:
        border = bops.multiContraction(C, border, '2', '0', cleanOr2=True)
        return border
    corner = getClosedCorner(AEnv, BEnv, border, width, height, projections)
    return corner


def getTopRightCorner(AEnv, BEnv, C, D, width, height, projections):
    border = tn.Node(np.eye(C[2].dimension, dtype=complex))
    for i in range(height):
        if i % 2 == 0:
            op = C
        else:
            op = D
        border = bops.multiContraction(border, op, [len(border.edges) - 1], '0', cleanOr1=True)
    for i in range(width):
        if i % 0 == 0:
            op = D
        else:
            op = C
        border = bops.multiContraction(op, border, '2', '0', cleanOr2=True)
    if height == 0:
        border = bops.multiContraction(border, D, [len(border.edges) - 1], '0', cleanOr1=True)
        return border
    if width == 0:
        border = bops.multiContraction(C, border, '2', '0', cleanOr2=True)
        return border
    corner = getClosedCorner(AEnv, BEnv, border, width, height, projections)

# TODO this is very wasteful since I create every corner from scratch,fix
def getMeasurement(A, B, AEnv, BEnv, C, D, width, height):
    projections = np.ones((height, width)) * -1
    # TODO handle edges
    for i in range(height):
        for j in range(width):
            topLeftCorner = getTopLeftCorner(AEnv, BEnv, C, D, width=j, height=i + 1, projections=projections)
            topRightCorner = getTopRightCorner(AEnv, BEnv, C, D, width=i, height=width - j,
                                             projections=np.transpose(projections[:i, j:])[::-1])
            bottomLeftCorner = getBottomLeftCorner(AEnv, BEnv, C, D, width=height - i - 1, height=j + 1,
                                               projections=np.transpose(projections[height:i:-1, :j + 1]))
            bottomRightCorner = getBottomRightCorner(AEnv, BEnv, C, D, width=width - j - 1, height=height - i,
                                                projections=projections[i:, j + 1:])
            b = 1

