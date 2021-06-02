import pickle
from matplotlib import pyplot as plt
import numpy as np
import os
import toricCode
import re
import basicAnalysis as ban

d = 2

rootdir = './results'
def findResults(n, N, opt='p'):
    regex = re.compile('organized_' + opt + str(n) + '_N_' + str(N) + '_*')
    regex2 = re.compile('organized_' + opt + '_' + str(n) + '_' + str(N) + '*')
    for root, dirs, files in os.walk(rootdir):
        for file in files:
            if regex.match(file) or regex2.match(file):
                return file


M = 1000
Ns = [4 * k for k in range(1, 7)]
colors = ['blueviolet', 'blue', 'deepskyblue', 'green', 'yellowgreen', 'orange']
vcolors = ['blueviolet', 'deepskyblue', 'green', 'orange']
legends = []
option = 'toric'
Vs = np.zeros(len(Ns))
ns = [1, 2, 3, 4]
p2s = []
for i in range(len(Ns)):
    p2s.append(toricCode.getPurity(2, (i + 1) * 2))
dops = True
if dops:
    for n in ns:
        precisions = []
        for i in range(len(Ns)):
            N = Ns[i]
            spaceSize = d**N
            if n == 2:
                legends.append(r'$N_A = ' + str(N) + '$')
            with open('./results/organized_' + option + '_' + str(n) + '_' + str(N), 'rb') as f:
                organized = np.array(pickle.load(f))
            # plt.plot(organized)
            # plt.title(str(n) + ' ' + str(N))
            # plt.show()
            p2 = p2s[i]
            expected = p2**(n-1)
            # numOfExperiments = 10
            # numOfMixes = 20
            # precision = np.zeros(int(len(organized) / numOfExperiments))
            # for mix in range(numOfMixes):
            #     np.random.shuffle(organized)
            #     for j in range(1, int(len(organized)/numOfExperiments)):
            #         currPrecision= np.average([np.abs(np.average( \
            #             organized[c * int(len(organized)/numOfExperiments):c * int(len(organized)/numOfExperiments)+j]) - expected) \
            #                                    for c in range(numOfExperiments)])
            #         if mix == 0:
            #             precision[j] = currPrecision
            #         else:
            #             precision[j] = (precision[j] * mix + currPrecision) / (mix + 1)
            # with open('results/precision_N_' + str(N) + '_n_' + str(n), 'rb') as f:
            #     precision = pickle.load(f)
            # axs[n-2].plot([(m * M + M - 1) / (varianceNormalizations[n - 2] ** N) for m in range(len(precision) - 1)],
            #          precision[1:] / expected, color=colors[i])
            variance = np.sum(np.abs(organized - np.average(organized))**2) / (len(organized) - 1)
            Vs[i] = np.real(variance / expected**2)
        Vs = Vs * M
        ban.linearRegression(Ns, Vs + 1, vcolors[n - 2], r'$p_' + str(n) + '$')


doR3 = False
if doR3:
    n = 3
    for i in range(len(Ns)):
        N = Ns[i]
        m = M - 1
        # estimation = []
        # organized = []
        # with open('./results/' + str(findResults(n, N)), 'rb') as f:
        #     organized = np.array(pickle.load(f))
        #     organized = organized[organized < 50]
        p2 = p2s[i]
        expected = p2 ** (n - 1)
        #     numOfExperiments = 10
        #     numOfMixes = 10
        #     precision = np.zeros(int(len(organized) / numOfExperiments))
        #     for mix in range(numOfMixes):
        #         np.random.shuffle(organized)
        #         for j in range(1, int(len(organized)/numOfExperiments)):
        #             currPrecision= np.average([np.abs(np.average( \
        #                 organized[c * int(len(organized)/numOfExperiments):c * int(len(organized)/numOfExperiments)+j]) - expected) \
        #                                        for c in range(numOfExperiments)])
        #             if mix == 0:
        #                 precision[j] = currPrecision
        #             else:
        #                 precision[j] = (precision[j] * mix + currPrecision) / (mix + 1)
        # with open('results/precision_N_' + str(N) + '_Rn_' + str(n), 'rb') as f:
        #     precision = pickle.load(f)
        # axs[3].plot([(m * M + M - 1) / 1.56**N for m in range(len(precision) - 1)],
        #          precision[1:] / expected, color=colors[i])
        # axs[3].set_yscale('log')
        # axs[3].set_ylabel(r'$\frac{|R_' + str(n) + ' - \\mathrm{est}|}{R_' + str(n) + '}$', fontsize=18)

# plt.xlabel(r'$M/(V^{N_A})$', fontsize=16)
# legend = plt.legend(legends, loc=0,
#            bbox_to_anchor=(0.25, 4), edgecolor='black')
# legend.get_frame().set_alpha(None)
# legend.get_frame().set_facecolor((1, 1, 1, 1))
# plt.subplots_adjust(left=0.15, bottom=0.15)
plt.show()
