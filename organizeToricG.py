import pickle
import numpy as np
from matplotlib import pyplot as plt

gs = [np.round(0.1 * k, 1) for k in range(1, 11)]

avgs = []
yerrs = []
for g in gs:
    with open('toricG/organized_g_' + str(g) + '_2_16', 'rb') as f:
        org = np.array(pickle.load(f)) / 1000
    with open('toricG/conserved_g_' + str(g) + '_2_16', 'rb') as f:
        con = np.array(pickle.load(f)) / 1000
    avg = np.average(org)
    var = np.sum(np.abs(org - avg) ** 2) / (len(org) - 1)
    yerrs.append(np.sqrt(var / len(org)))
    avgs.append(avg)
# plt.plot(gs, avgs)
plt.xlabel(r'$g$')
plt.ylabel(r'$p_2$')
plt.errorbar(gs, avgs, yerr=np.array(yerrs))
plt.show()