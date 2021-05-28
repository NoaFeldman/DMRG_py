import pickle
import numpy as np
from matplotlib import pyplot as plt

gs = [np.round(0.1 * k, 1) for k in range(1, 11)] + [np.round(0.15 + 0.1 * k, 2) for k in range(5)]
gs = np.sort(gs)

options = ['A', 'B', 'B', 'AB', 'AC', 'BC', 'ABC']
signs = [1, 1, 1, -1, -1, -1, 1]

n = 2
topoRenyis = np.zeros(len(gs))
yerrs = np.zeros(len(gs))
for j in range(len(options)):
    optrenyis = []
    optErrs = []
    option = options[j]
    for i in range(len(gs)):
        g = gs[i]
        with open('toricG/organized_g_' + str(g) + '_' + option + '_2_4', 'rb') as f:
            org = np.array(pickle.load(f)) / 1000
        with open('toricG/conserved_g_' + str(g) + '_' + option + '_2_4', 'rb') as f:
            con = np.array(pickle.load(f)) / 1000
        avg = np.average(org)
        sampleVar = np.sum(np.abs(org - avg) ** 2) / (len(org) - 1)
        standardDev = np.sqrt(sampleVar / len(org))
        renyi = 1 / (1 - n) * np.log(avg)
        topoRenyis[i] += signs[j] * renyi
        renyiErr = standardDev / avg
        yerrs[i] = np.sqrt(np.abs(yerrs[i])**2 + np.abs(renyiErr)**2)
        optrenyis.append(renyi)
        optErrs.append(renyiErr)
    # plt.errorbar(gs, optrenyis, yerr=optErrs)
    # plt.title(option)
    # plt.show()
plt.errorbar(gs, topoRenyis, yerr=yerrs)
plt.xlabel(r'string tension $g$', fontsize=16)
plt.ylabel(r'2nd topological Renyi $S^{(2)}_\gamma$', fontsize=16)
plt.show()
