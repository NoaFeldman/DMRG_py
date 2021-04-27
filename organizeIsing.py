import pickle
import numpy as np
from matplotlib import pyplot as plt

hfs = [0.4 * k for k in range(6)] + [2.6 + 0.1 * k for k in range(10)] + [3.8 + 0.4 * k for k in range(4)]

avgs = []
for hf in hfs:
    hf = np.round(hf, 1)
    if hf == int(hf):
        hf = int(hf)
    with open('ising/organized_hf_' + str(hf) + '_2_16', 'rb') as f:
        org = np.array(pickle.load(f)) / 1000
    org = org[org < 40]
    with open('ising/conserved_hf_' + str(hf) + '_2_16', 'rb') as f:
        con = np.array(pickle.load(f)) / 1000
    # plt.plot(np.real(org))
    # plt.plot(np.imag(org))
    # plt.title(str(hf))
    # plt.show()
    avg = np.average(org)
    var = np.sum(np.abs(org - avg) ** 2) / (len(org) - 1)
    print(hf, avg, var, var / len(org))
    avgs.append(avg)
plt.plot(hfs, avgs)
plt.xlabel(r'$h$')
plt.ylabel(r'$p_2$')
plt.scatter(hfs, avgs)
plt.show()