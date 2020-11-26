import numpy as np
import basicOperations as bops
import tensornetwork as tn
import scipy
import pickle
from datetime import datetime
import statistics


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
projs = [proj0Tensor, proj1Tensor]
def getP(d, s, us, estimateFunc, arguments):
    currUs = [tn.Node(np.eye(d)) for i in range(len(us))]
    for i in range(len(us)):
        currUs[i].tensor = np.matmul(np.matmul(us[i], projs[int(s & d ** i > 0)]), np.conj(np.transpose(us[i])))
    result = wrapper(estimateFunc, arguments + [currUs])
    bops.removeState(currUs)
    return result



# Based on my analysis in unitary_alternatives.
# We eventually calculate for each site M_i \rho_ij G_j M_k \rho_kl G_l
# for real Gaussian M, G.
# Averaging we get \delta_il\delta_jk
# This function returns mat_ij = M_iG_j
def getNonUnitaryRandomOp(d, randOption):
    if randOption == 'complex':
        M = (np.random.randint(2, size=d) * 2 - 1 + 1j * (np.random.randint(2, size=d) * 2 - 1)) / np.sqrt(2)
        G = (np.random.randint(2, size=(1, d)) * 2 - 1 + 1j * (np.random.randint(2, size=(1, d)) * 2 - 1)) / np.sqrt(2)
    elif randOption == 'real':
        M = np.random.randint(2, size=d) * 2 - 1
        G = np.random.randint(2, size=(1, d)) * 2 - 1
    elif randOption == 'gaussian':
        M = np.random.randn(2)
        G = np.random.randn(1, 2)
    elif randOption == 'unitCircle':
        M = np.exp(1j * 2 * np.pi * np.random.uniform(size=d))
        G = np.exp(1j * 2 * np.pi * np.random.uniform(size=(1, d)))
    elif randOption == 'slice8':
        M = np.exp(1j * 0.25 * np.pi * np.random.randint(8, size=d))
        G = np.exp(1j * 0.25 * np.pi * np.random.randint(8, size=(1, d)))
    return tn.Node(np.kron(M, np.transpose(G)))

def localNonUnitaries(N, M, randOption, estimateFunc, arguments, filename, d=2):
    start = datetime.now()
    avg = 0
    for m in range(int(M * d**N)):
        ops = [getNonUnitaryRandomOp(d, randOption) for i in range(N)]
        expectation = wrapper(estimateFunc, arguments + [ops])
        estimation = np.abs(expectation)**2
        mc = m % M
        avg = (avg * mc + estimation) / (mc + 1)
        if m % M == M - 1:
            with open(filename + '_N_' + str(N) + '_' + randOption + '_M_' + str(M) + '_m_' + str(m), 'wb') as f:
                pickle.dump(avg, f)
                print(avg)
                avg = 0
        bops.removeState(ops)
    end = datetime.now()
    with open(filename + '_time_N_' + str(N) + '_M_' + str(M), 'wb') as f:
        pickle.dump((end - start).total_seconds(), f)


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