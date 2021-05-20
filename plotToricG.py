import pickle
from matplotlib import pyplot as plt
import numpy as np
import basicOperations as bops
import tensornetwork as tn

d = 2
gs = [np.round(0.1 * k, 1) for k in range(1, 11)]

def sampleAvgVariance(organized):
    avg = np.average(organized)
    var = np.sum(np.abs(organized - avg) ** 2) / (len(organized) - 1)
    return avg, var


opts = ['16', '8', '24_exc16', '24_exc18', '24']
signs = [1, 1, 1, -1, -1, -1, 1]
estimations = np.zeros(len(gs))
vars = []
for i in range(len(gs)):
    g = gs[i]
    for j in range(len(opts)):
        opt = opts[j]
        with open('toricG/organized_g_' + str(g) + '_2_' + opt, 'rb') as f:
            organized = np.array(pickle.load(f)) / 1000
        with open('toricG/conserved_g_' + str(g) + '_2_' + opt, 'rb') as f:
            converged = np.array(pickle.load(f)) / 1000
            # plt.plot(converged)
            # plt.show()
        avg, var = sampleAvgVariance(organized)
        estimations[i] += avg * signs[j]
        vars.append(np.sqrt(var / len(organized)))
plt.plot(gs, estimations)
plt.show()

gs = [np.round(0.1 * k, 1) for k in range(1, 11)]
p2s = [0] * len(gs)
for i in range(len(gs)):
    g = np.round(gs[i], 1)
    with open('toricG/toricBoundaries_g_' + str(g), 'rb') as f:
        [upRow, downRow, leftRow, rightRow, openA, openB, A, B] = pickle.load(f)
    circle = bops.multiContraction(
        bops.multiContraction(bops.multiContraction(upRow, rightRow, '3', '0'), upRow, '5', '0'), leftRow, '70',
        '03')

    openA = tn.Node(
        np.transpose(np.reshape(np.kron(A.tensor, np.conj(A.tensor)), [d ** 2, d ** 2, d ** 2, d ** 2, d, d]),
                     [4, 0, 1, 2, 3, 5]))
    openB = tn.Node(
        np.transpose(np.reshape(np.kron(B.tensor, np.conj(B.tensor)), [d ** 2, d ** 2, d ** 2, d ** 2, d, d]),
                     [4, 0, 1, 2, 3, 5]))
    ABNet = bops.permute(
        bops.multiContraction(bops.multiContraction(openB, openA, '2', '4'),
                              bops.multiContraction(openA, openB, '2', '4'), '28', '16',
                              cleanOr1=True, cleanOr2=True),
        [1, 5, 6, 13, 14, 9, 10, 2, 0, 4, 8, 12, 3, 7, 11, 15])
    dm = bops.multiContraction(circle, ABNet, '01234567', '01234567')
    ordered = np.round(np.reshape(dm.tensor, [16, 16]), 14)
    ordered /= np.trace(ordered)
    p2s[i] = np.trace(np.matmul(ordered, ordered))
    estimations[i] += 2 * p2s[i]
plt.plot(gs, estimations)
plt.show()
