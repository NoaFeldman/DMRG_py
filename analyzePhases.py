import pickle
import numpy as np
from matplotlib import pyplot as plt

dir = 'ising/'
thetas = [0.1 * k for k in range(6)]
phis = [0.1 * k for k in range(6)]
hf = 5

results = np.zeros((len(thetas), len(phis)))
for i in range(len(thetas)):
    theta = np.round(thetas[i], 1)
    for j in range(len(phis)):
        phi = np.round(phis[j], 1)
        with open(dir + 'organized_ising_' + str(hf) + '_t_' + str(theta) + '_p_' + str(phi) + '_2_16', 'rb') as f:
            organized = np.array(pickle.load(f)) / 1000
        vars = np.zeros(len(organized))
        for k in range(2, len(organized)):
            avg = np.average(organized[:k])
            var = np.sum(np.abs(organized[:k] - avg)**2 / (k - 1))
            vars[k] = var
        # plt.plot(vars)
        # plt.show()
        results[i, j] = var
plt.pcolormesh(results)
plt.colorbar()
plt.show()