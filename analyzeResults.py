import pickle
from matplotlib import pyplot as plt
import numpy as np


M = 1000
Ns = [4, 8]
for N in Ns:
    result = [0] * N**2
    purity = 1 / (8 + 6 * (N - 4))
    for m in range(N**2):
        with open('results/toric_local_full_N_' + str(N) + '_M_' + str(M) + '_m_' + str(m * M + M - 1), 'rb') as f:
            result[m] = pickle.load(f)
    plt.plot([(m * M + M - 1) / N**2 for m in range(N**2)], np.abs(np.array(result) - purity) / purity)
    print(result)
plt.show()