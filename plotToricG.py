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


opts = ['A', 'B', 'C', 'AB', 'AC', 'BC', 'ABC']
signs = [1, 1, 1, -1, -1, -1, 1]
estimations = np.zeros(len(gs))
vars = []
n = 2
for i in range(len(gs)):
    g = gs[i]
    for j in range(len(opts)):
        opt = opts[j]
        with open('toricG/organized_g_' + str(g) + '_' + opt + '_2_4', 'rb') as f:
            organized = np.array(pickle.load(f)) / 1000
        with open('toricG/conserved_g_' + str(g) + '_' + opt + '_2_4', 'rb') as f:
            converged = np.array(pickle.load(f)) / 1000
            # plt.plot(converged)
            # plt.show()
        avg, var = sampleAvgVariance(organized)
        estimations[i] += 1 / (1 - n) * np.log(avg) * signs[j]
        vars.append(np.sqrt(var / len(organized)))

plt.plot(gs, np.abs(estimations))
plt.show()
