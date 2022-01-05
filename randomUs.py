import numpy as np
import basicOperations as bops
import tensornetwork as tn
import scipy
import pickle
from datetime import datetime
import gc

"""A Random matrix distributed with Haar measure"""
def haar_measure(n):
    z = (np.random.randn(n, n) + 1j * np.random.randn(n, n)) / np.sqrt(2.0)
    q,r = scipy.linalg.qr(z)
    d = np.diagonal(r)
    ph = d / np.absolute(d)
    q = np.multiply(q,ph,q)
    return q


def layers(l, d=16, numOfLayers=2):
    U = np.eye(d**l)
    for i in list(range(l - 1)) + list(range(l-3, -1, -1)):
        U = np.matmul(U, np.kron(np.eye(d**i), np.kron(haar_measure(d**2), np.eye(d**(l - (i  +2))))))
    return U


def nearestNeighborsCUE(N, d=2):
    res = np.eye(d**N)
    for i in list(range(N - 1)) + list(range(N-3, -1, -1)):
        res = np.matmul(res, np.kron(np.eye(d**i), np.kron(haar_measure(d**2), np.eye(d**(N - i - 2)))))
    return res


# create a global unitary from 2 layers of nearest neighbor unitaries
def globalUnitary(N, d=2, numberOfLayers=2):
    U = np.eye(d**N)
    for i in range(numberOfLayers):
        U = np.matmul(U, nearestNeighborsCUE(N, d))
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


def wrapper(func, args): # without star
    return func(*args)


proj0Tensor = np.zeros((2, 2), dtype=complex)
proj0Tensor[0, 0] = 1
proj1Tensor = np.zeros((2, 2), dtype=complex)
proj1Tensor[1, 1] = 1
projs =  [proj0Tensor, proj1Tensor]
def getP(d, s, us, estimateFunc, arguments):
    currUs = [tn.Node(np.eye(d)) for i in range(len(us))]
    for i in range(len(us)):
        currUs[i].tensor = np.matmul(np.matmul(us[i], projs[int(s & d ** i > 0)]), np.conj(np.transpose(us[i])))
    result = wrapper(estimateFunc, arguments + [currUs])
    bops.removeState(currUs)
    return result


def getUTheta(theta, d):
    if d == 2:
        u = np.zeros((d, d))
        u[0, 0] = np.cos(theta)
        u[0, 1] = np.sin(theta)
        u[1, 0] = -np.sin(theta)
        u[1, 1] = np.cos(theta)
        return u

def getUPhi(phi, d):
    if d == 2:
        u = np.zeros((d, d), dtype=complex)
        u[0, 0] = np.cos(phi)
        u[0, 1] = 1j * np.sin(phi)
        u[1, 0] = 1j * np.sin(phi)
        u[1, 1] = np.cos(phi)
        return u


def getUEta(eta, d):
    if d == 2:
        u = np.array([[np.exp(1j * eta), 0], [0, np.exp(-1j * eta)]])
        return u



def getVecsPool(d=2, random_option='full'):
    vecsPool = [np.array(arr) for arr in [[1, 1], [1, -1], [1, 1j], [1, -1j]]]
    if random_option == 'full':
        vecsPool += [np.array([np.sqrt(2), 0]), np.array([0, np.sqrt(2)])]
    vecsPool = vecsPool + [-arr for arr in vecsPool] + [1j * arr for arr in vecsPool] + [-1j * arr for arr in vecsPool]
    return vecsPool


def getNonUnitaryRandomOps(d, n, N, direction=0):
    vecsPool = getVecsPool()
    vecs = [[vecsPool[np.random.randint(len(vecsPool))] for i in range(N)] for j in range(n)]
    res = [[tn.Node(np.outer(vecs[j][i], np.conj(vecs[(j+1) % n][i]))) for i in range(N)] for j in range(n)]
    return res


def renyiEntropy(n, w, h, M, estimateFunc, arguments, filename, d=2, excludeIndices=[]):
    avg = 0
    N = w * h
    for m in range(M * 2**N * 10):
        ops = getNonUnitaryRandomOps(d, n, N)
        for ind in excludeIndices:
            for copy in range(len(ops[0])):
                ops[copy][ind] = [tn.Node(np.eye(d, dtype=complex)) for i in range(n)]
        estimation = 1
        for i in range(n):
            expectation = wrapper(estimateFunc, arguments + [ops[i]])
            estimation *= expectation
        avg += estimation
        if m % M == M - 1:
            with open(filename + '_n_' + str(n) + '_w_' + str(w) + '_h_' + str(h) + '_M_' + str(M) + '_m_' + str(m), 'wb') as f:
                pickle.dump(avg / M, f)
                avg = 0
            gc.collect()


def renyiNegativity(n, N, M,estimateFunc, arguments, filename, d=2):
    avg = 0
    for m in range(int(M * d ** N)):
        ops = [getNonUnitaryRandomOps(d, n, direction=int(N / 2 > i)) for i in range(N)]
        estimation = 1
        for i in range(n):
            expectation = wrapper(estimateFunc, arguments + [[op[i] for op in ops]])
            estimation *= expectation
        mc = m % M
        avg = (avg * mc + estimation) / (mc + 1)
        if m % M == M - 1:
            with open(filename + 'neg_n_' + str(n) + '_N_' + str(N) + '_M_' + str(M) + '_m_' + str(m), 'wb') as f:
                pickle.dump(avg, f)
                print(np.real(np.round(avg, 16)))
                avg = 0
        for op in ops:
            bops.removeState(op)


def exactPurity(l, xRight, xLeft, upRow, downRow, A, filename, d=2):
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
    with open(filename + '_l_' + str(l), 'wb') as f:
        pickle.dump(purity, f)
    return purity


def getPairUnitary(d):
    return tn.Node(np.reshape(haar_measure(d ** 2), [d] * 4))

##### Deprecated below #####

def localUnitariesFull(N, M, estimateFunc, arguments, filename, d=2):
    start = datetime.now()
    avg = 0
    for m in range(int(M * N**2)):
        us = [haar_measure(d) for i in range(N)]
        ps = [0] * d**N
        purity = 0
        for s in range(d**N):
            ps[s] = getP(d, s, us, estimateFunc, arguments)
        for s in range(d**N):
            for sp in range(d**N):
                purity += d**N * (-d)**(-localDistance(s, sp)) * ps[s] * ps[sp]
        avg = (avg * m + purity) / (m + 1)
        if m % M == M - 1:
            with open(filename + '_N_' + str(N) + '_M_' + str(M) + '_m_' + str(m), 'wb') as f:
                pickle.dump(avg, f)
    end = datetime.now()
    with open(filename + '_time_N_' + str(N) + '_M_' + str(M), 'wb') as f:
        pickle.dump((end - start).total_seconds(), f)


def mcStep(s, d, N, us, estimateFunc, arguments, probabilities):
    # flip one random spin
    newS = s ^ d ** (np.random.randint(N))
    if newS not in probabilities.keys():
        probabilities[newS] = getP(d, newS, us, estimateFunc, arguments)
    takeStep = np.random.rand() < probabilities[newS] / probabilities[s]
    if takeStep:
        s = newS
    return s


def localUnitariesMC(N, M, estimateFunc, arguments, filename, chi, d=2):
    start = datetime.now()
    avg = 0
    for m in range(int(M * N**2)):
        us = [haar_measure(d) for i in range(N)]
        probabilities = {}
        estimation = 0

        s = np.random.randint(0, 2**N)
        probabilities[s] = getP(d, s, us, estimateFunc, arguments)
        sp = np.random.randint(0, 2**N)
        probabilities[sp] = getP(d, sp, us, estimateFunc, arguments)
        for j in range(chi * N**2):
            estimation += d**N * (-d)**(-localDistance(s, sp))
            s = mcStep(s, d, N, us, estimateFunc, arguments, probabilities)
            sp = mcStep(sp, d, N, us, estimateFunc, arguments, probabilities)
        estimation /= (chi * N **2)

        avg = (avg * m + estimation) / (m + 1)
        if m % M == M - 1:
            with open(filename + '_N_' + str(N) + '_M_' + str(M) + '_m_' + str(m) + '_chi_' + str(chi), 'wb') as f:
                pickle.dump(avg, f)
    end = datetime.now()
    with open(filename + '_time_N_' + str(N) + '_M_' + str(M) + '_chi_' + str(chi), 'wb') as f:
        pickle.dump((end - start).total_seconds(), f)

