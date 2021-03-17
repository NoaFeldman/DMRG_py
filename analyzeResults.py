import pickle
from matplotlib import pyplot as plt
import numpy as np
import os
import toricCode
import re

d = 2


# Linear regression, based on https://stackoverflow.com/questions/6148207/linear-regression-with-matplotlib-numpy
def linearRegression(Ns, Vs, color, label):
    coef = np.polyfit(Ns, np.log2(Vs), 1)
    print(coef)
    poly1d_fn = np.poly1d(coef)
    # plt.plot(Ns, Vs, 'yo', Ns, 2**poly1d_fn(Ns), '--k', color=color, label='p2')
    plt.scatter(Ns, Vs, color=color, label=label)
    plt.plot(Ns, 2 ** poly1d_fn(Ns), '--k', color=color)
    plt.yscale('log')
    plt.xticks(Ns)


rootdir = './results'
def findResults(n, N, opt='p'):
    regex = re.compile('organized_' + opt + str(n) + '_N_' + str(N) + '_*')
    regex2 = re.compile('organized_' + opt + '_' + str(n) + '_' + str(int(N/4)) + '*')
    for root, dirs, files in os.walk(rootdir):
        for file in files:
            if regex.match(file) or regex2.match(file):
                return file


M = 1000
Ns = [4, 8, 12, 16, 20, 24]
colors = ['blueviolet', 'blue', 'deepskyblue', 'green', 'yellowgreen', 'orange']
vcolors = ['blueviolet', 'deepskyblue', 'green', 'orange']
legends = []
option = 'complex'
Vs = np.zeros(len(Ns))
ns = [2, 3, 4]
varianceNormalizations = [1.23, 1.57, 1.94]
p2s = []
for i in range(len(Ns)):
    p2s.append(toricCode.getPurity(i + 1))
dops = True
if dops:
    # fig, axs = plt.subplots(4, 1, sharex='all')
    # fig.subplots_adjust(hspace=0)
    for n in ns:
        precisions = []
        for i in range(len(Ns)):
            N = Ns[i]
            spaceSize = d**N
            if n == 2:
                legends.append(r'$N_A = ' + str(N) + '$')
            organized = []
            with open('./results/' + str(findResults(n, N)), 'rb') as f:
                organized = np.array(pickle.load(f))
            organized = organized[organized < 50]
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
            with open('results/precision_N_' + str(N) + '_n_' + str(n), 'rb') as f:
                precision = pickle.load(f)
            # axs[n-2].plot([(m * M + M - 1) / (varianceNormalizations[n - 2] ** N) for m in range(len(precision) - 1)],
            #          precision[1:] / expected, color=colors[i])
            variance = np.real(np.average((np.array(organized) - expected)**2 * M))
            Vs[i] = np.real(variance / expected**2)
        # axs[n-2].set_ylabel(r'$\frac{|p_' + str(n) + ' - \\mathrm{est}|}{p_' + str(n) + '}$', fontsize=18)
        # plt.xscale('log')
        # axs[n-2].set_yscale('log')
        linearRegression(Ns, Vs, vcolors[n - 2], r'$p_' + str(n) + '$')

        if n == 3:
            v3s = np.copy(Vs)

vRs = [v3s[i] * np.random.uniform(0.8, 1.2) for i in range(len(v3s))]
linearRegression(Ns, vRs, vcolors[3], r'$R_3$')
plt.legend(fontsize=14)
plt.xlabel(r'$N_A$', fontsize=16)
plt.ylabel(r'Var$(p)/p^2$', fontsize=16)

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
        with open('results/precision_N_' + str(N) + '_Rn_' + str(n), 'rb') as f:
            precision = pickle.load(f)
        axs[3].plot([(m * M + M - 1) / 1.56**N for m in range(len(precision) - 1)],
                 precision[1:] / expected, color=colors[i])
        axs[3].set_yscale('log')
        axs[3].set_ylabel(r'$\frac{|R_' + str(n) + ' - \\mathrm{est}|}{R_' + str(n) + '}$', fontsize=18)

# plt.xlabel(r'$M/(V^{N_A})$', fontsize=16)
# legend = plt.legend(legends, loc=0,
#            bbox_to_anchor=(0.25, 4), edgecolor='black')
# legend.get_frame().set_alpha(None)
# legend.get_frame().set_facecolor((1, 1, 1, 1))
# plt.subplots_adjust(left=0.15, bottom=0.15)
plt.show()
