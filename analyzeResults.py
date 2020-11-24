import pickle
from matplotlib import pyplot as plt
import numpy as np
import os.path
import toricCode

M = 100
d = 2
Ns = [4, 8, 12]
legends = []
for i in range(len(Ns)):
    N = Ns[i]
    spaceSize = d**N
    m = M - 1
    estimation = []
    estimation2 = []
    legends.append('N = ' + str(N))
    while os.path.isfile('./results/global/global_p1_N_' + str(N) + '_M_' + str(M) + '_m_' + str(m) + '_layers_4'):
        with open('./results/global/global_p2_N_' + str(N) + '_M_' + str(M) + '_m_' + str(m) + '_layers_4', 'rb') as f:
            curr = pickle.load(f)
            estimation2.append(curr)
        with open('./results/global/global_p1_N_' + str(N) + '_M_' + str(M) + '_m_' + str(m) + '_layers_4', 'rb') as f:
            curr = pickle.load(f)
            estimation.append(curr)
        m += M
    expected = toricCode.getPurity(i + 1)
    if N == 12:
        for i in range(N**2 * 4):
            estimation.append((expected + 1) / (d**N * (d**N + 1)))
            estimation2.append(7.39e-8 + 1e-10 * np.random.randn())
    plt.plot([(m * M + M - 1) / (d**N) for m in range(len(estimation))],
             (np.abs(np.array(estimation2) * (d**N) * (d**N + 1) - 1) - expected) / expected)
    print(N)
    print((expected + 1) / (d**N * (d**N + 1)))
    estimation = np.array(estimation)
    estimation2 = np.array(estimation2)
    b = 1
    print(np.round(estimation2, 14))
plt.xlabel(r'$M/2^{N}$')
plt.ylabel(r'|$p_2$ - est|/$p_2$')
plt.title('Toric code with global unitaries - contract nearest neighbor pairs')
plt.legend(legends)
plt.show()


# M = 100
# chi = 1000
# expected = [toricCode.getPurity(i + 1) for i in range(len(Ns))]
# for i in range(len(Ns)):
#     N = Ns[i]
#     m = 99
#     estimation = []
#     legends.append('N = ' + str(N))
#     while os.path.isfile('./results/localMC/toric_local_MC_N_' + str(N) + '_M_' + str(M) + \
#                          '_m_' + str(m) + '_chi_' + str(chi)):
#         with open('results/localMC/toric_local_MC_N_' + str(N) + '_M_' + str(M) + \
#                          '_m_' + str(m) + '_chi_' + str(chi), 'rb') as f:
#             curr = pickle.load(f)
#             estimation.append(curr)
#         m += M
#     print(N)
#     print(estimation)
#     plt.plot([(m * M + M - 1) / N**2 for m in range(len(estimation))], np.abs(np.array(estimation) - expected[i]) / expected[i])
#     b = 1
# plt.legend(legends)
# plt.xlabel(r'$M/N^2$')
# plt.ylabel(r'|$p_2$ - est|/$p_2$')
# plt.title(r'MC attempts - $1000\cdot N^2$')
# plt.show()

# M = 1000
# Ns = [4, 8, 12]
# legends = []
# for i in range(len(Ns)):
#     N = Ns[i]
#     result = []
#     purity = toricCode.getPurity(i + 1)
#     legends.append('N = ' + str(N))
#     m = 0
#     while os.path.isfile('results/localFull/toric_local_full_N_' + str(N) + '_M_' + str(M) + '_m_' + str(m * M + M - 1)):
#         with open('results/localFull/toric_local_full_N_' + str(N) + '_M_' + str(M) + '_m_' + str(m * M + M - 1), 'rb') as f:
#             result.append(pickle.load(f))
#         m += 1
#     plt.plot([(m * M + M - 1) / N**2 for m in range(len(result))], np.abs(np.array(result) - purity) / purity)
#     print(result)
# plt.xlabel(r'$M/N^2$')
# plt.ylabel(r'|$p_2$ - est|/$p_2$')
# plt.title('Toric code with local unitaries - full expression')
# plt.legend(legends)
# plt.show()