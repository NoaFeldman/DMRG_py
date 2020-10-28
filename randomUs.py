import numpy as np
import basicOperations as bops
import tensornetwork as tn
import scipy
import pickle
import ising
import time


def myGUE(n):
    z = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    for i in range(n):
        for j in range(i):
            z[i, j] = np.conj(z[j, i])
        z[i, i] = np.sqrt(2) * np.real(z[i, i])
    # z = (z + np.conj(np.transpose(z))) / 2
    return z / np.sqrt(2)

def nearestNeighborsGUE(N, d=2):
    res = np.eye(d**N)
    for i in range(N - 1):
        res = np.matmul(res, np.kron(np.eye(d**i), np.kron(myGUE(d**2), np.eye(d**(N - i - 2)))))
    return res



M = 100000
N = 6
n = 2
res = np.zeros((n, n), dtype=complex)
for i in range(M):
    g = myGUE(n)
    res += g*np.transpose(g) / M
b = 1
tests = [[1, 1, 1, 1],
         [0, 0, 0, 0],
         [2, 15, 15, 2],
         [0, 2, 0, 2],
         [1, 3, 2, 1]]
testResults = [0] * len(tests)
for i in range(M):
    G = nearestNeighborsGUE(N)
    for t in range(len(tests)):
        testResults[t] += G[tests[t][0], tests[t][1]] * G[tests[t][2], tests[t][3]]
testResults = [r / M for r in testResults]
b = 1


"""A Random matrix distributed with Haar measure"""
def haar_measure(n):
    z = (np.random.randn(n, n) + 1j * np.random.randn(n, n)) / np.sqrt(2.0)
    q,r = scipy.linalg.qr(z)
    d = np.diagonal(r)
    ph = d / np.absolute(d)
    q = np.multiply(q,ph,q)
    return q


# create a global unitary from 2 layers of nearest neighbor unitaries
def globalUnitary(N, d, numberOfLayers=2):
    U = np.eye(d**N)
    for i in range(numberOfLayers):
        u01 = np.kron(haar_measure(d**2), np.eye(d**2, dtype=complex))
        u02 = np.reshape(
            np.transpose(np.reshape(np.kron(haar_measure(d ** 2), np.eye(d ** 2, dtype=complex)), [d] * 2 * N),
                         [0, 2, 1, 3, 4, 6, 5, 7]), [d ** N, d ** N])
        u23 = np.kron(np.eye(d**2, dtype=complex), haar_measure(d**2))
        u13 = np.reshape(np.transpose(np.reshape(np.kron(haar_measure(d**2), np.eye(d**2, dtype=complex)), [d] * 2 * N),
                                      [2, 0, 3, 1, 6, 4, 7, 5]), [d**N, d**N])
        U = np.matmul(U, np.matmul(u01, np.matmul(u02, np.matmul(u23, u13))))
    return U


def estimateOp(xRight, xLeft, upRow, downRow, A, ops):
    N = len(ops)
    curr = xLeft
    for i in range(int(N / 2)):
        closedA = tn.Node(np.trace(bops.multiContraction(ops[i * 2], A, '1', '0').tensor, axis1=0, axis2=5))
        closedB = tn.Node(np.trace(bops.multiContraction(ops[i * 2 + 1], A, '1', '0').tensor, axis1=0, axis2=5))
        closed = bops.permute(bops.multiContraction(closedA, closedB, '1', '3'), [0, 3, 2, 4, 1, 5])
        curr = bops.multiContraction(bops.multiContraction(bops.multiContraction(
            curr, upRow, '0', '0'), closed, '023', '201', cleanOr1=True), downRow, '034', '012', cleanOr1=True)
    return bops.multiContraction(curr, xRight, '012', '012').tensor


def localDistance(s, sp):
    return bin(s ^ sp).count("1")


proj0Tensor = np.zeros((2, 2), dtype=complex)
proj0Tensor[0, 0] = 1
proj1Tensor = np.zeros((2, 2), dtype=complex)
proj1Tensor[1, 1] = 1
projs = [proj0Tensor, proj1Tensor]
def getP(d, s, us, xRight, xLeft, upRow, downRow, A):
    currUs = [tn.Node(np.eye(d)) for i in range(len(us))]
    for i in range(len(us)):
        currUs[i].tensor = np.matmul(np.matmul(us[i], projs[int(s & d ** i > 0)]), np.conj(np.transpose(us[i])))
    return estimateOp(xRight, xLeft, upRow, downRow, A, currUs)


def localUnitariesFull(l, M, A, xRight, xLeft, upRow, downRow, filename, d=2):
    start = time.time()
    avg = 0
    avgs = []
    for m in range(int(M * l**2)):
        t = estimateOp(xRight, xLeft, upRow, downRow, A, [tn.Node(np.eye(2)) for i in range(l)])
        xLeft = bops.multNode(xLeft, 1 / t)
        us = [haar_measure(d) for i in range(l)]
        ps = [0] * 2**l
        purity = 0
        for s in range(2**l):
            ps[s] = getP(d, s, us, xRight, xLeft, upRow, downRow, A)
        for s in range(2**l):
            for sp in range(2**l):
                purity += d**l * (-d)**(-localDistance(s, sp)) * ps[s] * ps[sp]
        avg = (avg * m + purity) / (m + 1)
        if m % M == M - 1:
            avgs.append(avg)
    end = time.time()
    with open(filename + '_l_' + str(l) + '_M_' + str(M), 'wb') as f:
        pickle.dump(avgs, f)
    with open(filename + '_time_l_' + str(l) + '_M_' + str(M), 'wb') as f:
        pickle.dump(end - start, f)


def localUnitariesMC(l, M, A, xRight, xLeft, upRow, downRow, filename, chi, d=2):
    start = time.time()
    avg = 0
    avgs = []
    for m in range(int(M * l**2)):
        t = estimateOp(xRight, xLeft, upRow, downRow, A, [tn.Node(np.eye(2)) for i in range(l)])
        xLeft = bops.multNode(xLeft, 1 / t)
        us = [haar_measure(d) for i in range(l)]
        probabilities = {}
        estimation = 0

        s = np.random.randint(0, 2**l)
        probabilities[s] = getP(d, s, us, xRight, xLeft, upRow, downRow, A)
        sp = np.random.randint(0, 2**l)
        probabilities[sp] = getP(d, sp, us, xRight, xLeft, upRow, downRow, A)
        for j in range(chi):
            estimation += d**l * (-d)**(-localDistance(s, sp))
            changeS = bool(np.random.randint(2))
            if changeS:
                # flip one random spin
                newS = s ^ d**(np.random.randint(l))
                if newS not in probabilities.keys():
                    probabilities[newS] = getP(d, newS, us, xRight, xLeft, upRow, downRow, A)
                newSP = sp
            else:
                newSP = sp ^ d**(np.random.randint(l))
                if newSP not in probabilities.keys():
                    probabilities[newSP] = getP(d, newSP, us, xRight, xLeft, upRow, downRow, A)
                newS = s
            takeStep = np.random.rand() < \
                   (probabilities[newS] * probabilities[newSP]) / \
                   (probabilities[s] * probabilities[sp])
            if takeStep:
                s = newS
                sp = newSP
        estimation /= chi

        avg = (avg * m + estimation) / (m + 1)
        if m % M == M - 1:
            avgs.append(avg)
    end = time.time()
    with open(filename + '_l_' + str(l) + '_M_' + str(M), 'wb') as f:
        pickle.dump(avgs, f)
    with open(filename + '_time_l_' + str(l) + '_M_' + str(M), 'wb') as f:
        pickle.dump(end - start, f)


def localUnitariesMC(l, M, A, xRight, xLeft, upRow, downRow, filename, chi, d=2):
    start = time.time()
    avg = 0
    avgs = []
    for m in range(int(M * l**2)):
        t = estimateOp(xRight, xLeft, upRow, downRow, A, [tn.Node(np.eye(2)) for i in range(l)])
        xLeft = bops.multNode(xLeft, 1 / t)
        us = [haar_measure(d) for i in range(l)]
        probabilities = {}
        estimation = 0
        s = np.random.randint(0, 2**l)
        probabilities[s] = getP(d, s, us, xRight, xLeft, upRow, downRow, A)
        for j in range(chi):
            sp = s
            for jp in range(chi):
                estimation += d**l * (-1)**(-localDistance(s, sp))
                # flip one random spin
                newSP = sp ^ d**(np.random.randint(l))
                if newSP not in probabilities.keys():
                    probabilities[newSP] = getP(d, newSP, us, xRight, xLeft, upRow, downRow, A)
                takeStep = np.random.rand() < \
                   (d**(-localDistance(s, newSP)) * probabilities[newSP]) / \
                   (d**(-localDistance(s, sp)) * probabilities[sp])
                if takeStep:
                    sp = newSP
            # flip one random spin
            newS = s ^ d**(np.random.randint(l))
            if newS not in probabilities.keys():
                probabilities[newS] = getP(d, newS, us, xRight, xLeft, upRow, downRow, A)
            takeStep = np.random.rand() < probabilities[newS] / probabilities[s]
            if takeStep:
                s = newS

        estimation /= chi**2

        avg = (avg * m + estimation) / (m + 1)
        if m % M == M - 1:
            avgs.append(avg)
    end = time.time()
    with open(filename + '_l_' + str(l) + '_M_' + str(M), 'wb') as f:
        pickle.dump(avgs, f)
    with open(filename + '_time_l_' + str(l) + '_M_' + str(M), 'wb') as f:
        pickle.dump(end - start, f)


def exactPurity(l, xRight, xLeft, upRow, downRow, A, filename, d=2):
    start = time.time()
    curr = xLeft
    pair = bops.permute(bops.multiContraction(A, A, '2', '4'), [1, 6, 3, 7, 2, 8, 0, 5, 4, 9])
    for i in range(int(l / 2)):
        curr = bops.multiContraction(bops.multiContraction(curr, upRow, '0', '0'), downRow, '1', '0')
        curr = bops.multiContraction(curr, pair, [0, i * 4 + 1, i * 4 + 2, i * 4 + 4, i * 4 + 5], '20145')
        curr = bops.permute(curr, [i * 4, i * 4 + 2, i * 4 + 1] + list(range(i * 2)) + [i * 4 + 3, i * 4 + 4] +
                            list(range(i * 2, i * 4)) + [i * 4 + 5, i * 4 + 6])
    dm = bops.multiContraction(curr, xRight, '012', '012')
    ordered = np.reshape(dm.tensor, [d**l, d**l]) / np.trace(np.reshape(dm.tensor, [d**l, d**l]))
    purity = sum(np.linalg.eigvalsh(np.matmul(ordered, ordered)))
    end = time.time()
    with open(filename + '_l_' + str(l), 'wb') as f:
        pickle.dump(purity, f)
    with open(filename + '_time_l_' + str(l), 'wb') as f:
        pickle.dump(end - start, f)
    return purity


M = 1e2
# exactPurity(2, ising.xRight, ising.xLeft, ising.upRow, ising.upRow, ising.A, 'exact')
# localUnitariesFull(2, M, ising.A, ising.xRight, ising.xLeft, ising.upRow, ising.upRow, 'localfull')
# localUnitariesMC(4, M, ising.A, ising.xRight, ising.xLeft, ising.upRow, ising.upRow, 'localMC', 100)
b = 1


