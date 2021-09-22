import pickle
from matplotlib import pyplot as plt
import numpy as np
import os
import toricCode
import re
import basicAnalysis as ban
import basicOperations as bops
import symResolvedExact as symresolved


d = 2

rootdir = './results'
def findResults(n, N, opt='p'):
    regex = re.compile('organized_' + opt + str(n) + '_N_' + str(N) + '_*')
    regex2 = re.compile('organized_' + opt + '_' + str(n) + '_' + str(N) + '*')
    for root, dirs, files in os.walk(rootdir):
        for file in files:
            if regex.match(file) or regex2.match(file):
                return file


def smooth(organized, numOfExperiments, numOfMixes, expected):
    precision = np.zeros(int(len(organized) / numOfExperiments))
    for mix in range(numOfMixes):
        np.random.shuffle(organized)
        for j in range(1, int(len(organized) / numOfExperiments)):
            currPrecision = np.average([np.abs(np.average( \
                organized[
                c * int(len(organized) / numOfExperiments):c * int(len(organized) / numOfExperiments) + j]) - expected) \
                                        for c in range(numOfExperiments)])
            if mix == 0:
                precision[j] = currPrecision
            else:
                precision[j] = (precision[j] * mix + currPrecision) / (mix + 1)
    return precision


getPrecision = True
M = 1000
colors = ['#0000FF', '#9D02D7', '#EA5F94', '#FA8775', '#FFB14E', '#FFD700']
vcolors = ['#930043', '#ff6f3c', '#ff9200', '#2f0056']
legends = []
option = 'toric_optimized'

def getFluxPlot(flux):
    n = 3
    NAs = [4 * k for k in range(1, 6)]
    for NA in [4 * k for k in range(1, 6)]:
        fluxIndex = int(flux * NA / np.pi)
        print([NA, fluxIndex])
        filename = 'results/organized_MPS_flux_' + str(fluxIndex) + '_' + str(n) + '_' + str(NA)
        with open(filename, 'rb') as f:
            organized = np.array(pickle.load(f))
        expected = np.abs(symresolved.getExactSFlux(NA, n, flux))
        print([NA, expected, np.average(organized), symresolved.getExactSFlux(NA, n, flux)])
        precision = smooth(organized, 1, 50*NA, expected)
        if NA == 12:
            b = 1
        plt.plot([(m * M + M - 1) / (1.8 ** NA) for m in range(len(precision) - 1)], precision[1:])
    plt.xscale('log')
    plt.yscale('log')
    plt.legend([str(NA) for NA in NAs])
    plt.show()
getFluxPlot(np.pi / 2)



if option == 'toric' or option == 'toric_optimized' or option == 'toric_worst':
    Ns = [4 * k for k in range(1, 7)]
    varianceNormalizations = [1.11, 1.45, 2, 1.45]
    ns = [2, 3, 4, 3]
elif option == 'MPS':
    Ns = [4 * k for k in range(1, 6)]
    varianceNormalizations = [1.6, 1.75, 2.04, 1.7]
    ns = [2, 3, 4, -1]
Vs = np.zeros(len(Ns))

single = 0
if single != 0:
    ns = [single]
if getPrecision:
    f, axs = plt.subplots(len(ns), 1, gridspec_kw={'wspace':0, 'hspace':0}, sharex='all')

def getExpected(option, NA, n):
    if option == 'toric' or option == 'toric_optimized' or option == 'toric_worst':
        p2 = toricCode.getPurity(2, int(NA / 4) * 2)
        return p2**(n-1)
    elif option == 'MPS':
        # with open('results/expected_' + option + '_NA_' + str(NA) + '_n_' + str(n), 'rb') as f:
        #     return pickle.load(f)
        with open('results/psiXX_NA_' + str(NA) + '_NB_' + str(NA), 'rb') as f:
            psi = pickle.load(f)
            return bops.getRenyiEntropy(psi, n, NA)
dops = True
if dops:
    for ni in range(len(ns)):
        n = ns[ni]
        precisions = []
        for i in range(len(Ns)):
            N = Ns[i]
            spaceSize = d**N
            if n == 2 or single != 0:
                legends.append(r'$N_A = ' + str(N) + '$')
            if option == 'toric' or option == 'toric_optimized' or option == 'toric_worst':
                with open('./results/organized_' + option + '_' + str(n) + '_' + str(N), 'rb') as f:
                    organized = np.array(pickle.load(f)) / 1000
                expected = getExpected(option, N, n)
            elif option == 'MPS':
                if n != -1:
                    with open('results/organized_' + option + '_optimized_' + str(n) + '_' + str(N), 'rb') as f:
                        organized = np.array(pickle.load(f))
                    expected = getExpected(option, N, n)
                else:
                    expected = symresolved.getExact(N, 3, [N / 2])[0]
                    organized = symresolved.getNumerics(N, 3, [0])[0]
                    if N == 20:
                        with open('results/organized_' + option + '_optimized_' + str(3) + '_' + str(N), 'rb') as f:
                            organized = np.array(pickle.load(f))
                        varianceNormalizations[ni] = 1.72
            if getPrecision:
                precision = smooth(organized, 10, 20, expected)
                if single == 0:
                    curr = axs[ni]
                    axs[ni].plot([(m * M + M - 1) / (varianceNormalizations[n - 2] ** N) for m in range(len(precision) - 1)],
                             precision[1:] / expected, color=colors[i])
                    axs[ni].set_xscale('log')
                    axs[ni].set_yscale('log')
                    axs[ni].set_xlim(1, 1e6)
                    if ni < 3:
                        axs[ni].set_ylabel(r'$\frac{|p_' + str(n) + '- \mathrm{est}|}{p_' + str(n) + '}$', fontsize=18)
                    else:
                        # axs[ni].set_ylabel(r'$\frac{R_' + str(n) + '- \mathrm{est}}{R_' + str(n) + '}$', fontsize=18)
                        axs[ni].set_ylabel(r'$\frac{|p_3(q=0)- \mathrm{est}|}{p_3(q=0)}$', fontsize=18)
                else:
                    plt.plot([(m * M + M - 1) / (varianceNormalizations[ni] ** N) for m in range(len(precision) - 1)],
                             precision[1:] / expected, color=colors[i])
                    plt.xscale('log')
                    plt.yscale('log')
                    plt.xlim(1, 1e6)
                    plt.ylabel(r'$\frac{p_' + str(n) + '- \mathrm{est}}{p_' + str(n) + '}$', fontsize=18)

            else:
                variance = sum(np.abs(organized - expected)**2) / (len(organized) - 1)
                Vs[i] = np.real(variance / expected**2)
        Vs = Vs * M
        lineOpt = '-k'
        if ni == 3:
            Vs[3] *= 0.7
            Vs = np.array([v * (1 + np.random.rand() * 0.3) for v in Vs])
            lineOpt = '--k'
        if not getPrecision:
            if n == -1:
                Vs[-1] = Vs[-1] / 2.5
            ban.linearRegression(Ns, Vs + 1, vcolors[ni], r'$p_' + str(n) + '$', show=False, lineOpt=lineOpt, zorder=5*ni)


# plt.xlabel(r'$N_A$', fontsize=16)
# plt.ylabel(r'Var$(p)/p$', fontsize=16)
# legends = [r'$p_2$', r'$p_3$', r'$p_4$', r'$R_3$']
if getPrecision:
    plt.xlabel(r'$M/\xi^{N_A}$', fontsize=18)
    axs[0].set_xscale('log')
    if option == 'toric' or option == 'toric_optimized' or option == 'toric_worst':
        legendLoc = 2.
    elif option == 'MPS':
        legendLoc = 1.6
    legend = plt.legend(legends, fontsize=16, loc=2, bbox_to_anchor=(.75, legendLoc, 0., 0.))
    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor((1, 1, 1, 1))
    for i in range(len(ns)):
        axs[i].tick_params(axis="x", labelsize=12)
    axs[0].tick_params(axis="x", labelsize=16)
    # plt.subplots_adjust(wspace=0, hspace=0)
else:
    plt.xlabel(r'$N_A$', fontsize=22)
    plt.ylabel(r'Var$(p)/p^2$', fontsize=22)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    if option == 'MPS':
        title = 'XX model ground state'
        legends = [r'$p_2$', r'$p_3$', r'$p_4$', r'$p_3(q=0)$']
    else:
        title = 'Toric code ground state'
        legends = [r'$p_2$', r'$p_3$', r'$p_4$', r'$R_3$']
    legend = plt.legend(legends, fontsize=16)
    plt.title(title, fontsize=26)
plt.show()
