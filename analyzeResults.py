import pickle
from matplotlib import pyplot as plt
import numpy as np


M = 100
Ns = [4, 8, 12, 16, 20]



M = 1000
Ns = [4, 8, 12]
Ms = [16, 64, 36]
expected = [1/8, 1/32, 1/128]
legends = []
for i in range(len(Ns)):
    N = Ns[i]
    result = [0] * Ms[i]
    purity = expected[i]
    legends.append('N = ' + str(N))
    for m in range(Ms[i]):
        # with open('results/toric_local_full_N_' + str(N) + '_M_' + str(M) + '_m_' + str(m * M + M - 1), 'rb') as f:
        with open('results/toric_local_MC_N_' + str(N) + '_M_' + str(M) + '_m_' + str(m * M + M - 1) \
                  + '_chi_100', 'rb') as f:
            result[m] = pickle.load(f)
    plt.plot([(m * M + M - 1) / N**2 for m in range(Ms[i])], np.abs(np.array(result) - purity) / purity)
    print(result)
plt.xlabel(r'$M/N^2$')
plt.ylabel(r'|$p_2$ - est|/$p_2$')
plt.legend(legends)
plt.show()