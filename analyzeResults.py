import pickle
from matplotlib import pyplot as plt
import numpy as np
import os
import toricCode
import re
import basicAnalysis as ban
import basicOperations as bops
import symResolvedExact as symresolved

plt.rcParams['font.family'] = 'Times New Roman'
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
# vcolors = ['#ff6f3c', '#FFD700', '#2f0056', '#930043', '#0000FF'] # toric
vcolors = ['#ff6f3c', '#EA5F94', '#930043'] # XX
vmarkers = ['o', '^', 'x']
vlineopts = ['-k', '--k', ':k']
legends = []
model = 'toric'

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
        precision = smooth(organized, 1, 1, expected)
        if NA == 12:
            b = 1
        plt.plot([(m * M + M - 1) / (1.8 ** NA) for m in range(len(precision) - 1)], precision[1:])
    plt.xscale('log')
    plt.yscale('log')
    plt.legend([str(NA) for NA in NAs])
    plt.show()
# getFluxPlot(np.pi / 2)


if model[:5] == 'toric':
    Ns = [4 * k for k in range(1, 7)]
    varianceNormalizations = [1.11, 1.45, 2, 1.45]
    ns = [2, 3, 4, 3]
elif model == 'XX':
    Ns = [4 * k for k in range(1, 6)]
    varianceNormalizations = [1.6, 1.75, 2.04, 1.72]
    ns = [2, 3, 4]
Vs = np.zeros(len(Ns))

single = 0
if single != 0:
    ns = [single]
if getPrecision:
    f, axs = plt.subplots(len(ns), 1, gridspec_kw={'wspace':0, 'hspace':0}, sharex='all')

def getExpected(model, NA, n):
    if model[:5] == 'toric':
        p2 = toricCode.getPurity(2, int(NA / 4) * 2)
        return p2**(n-1)
    elif model[:2] == 'XX' or model[:3] == 'MPS':
        if n != -1:
            # with open('results/expected_' + model + '_NA_' + str(NA) + '_n_' + str(n), 'rb') as f:
            #     return pickle.load(f)
            with open('results/psiXX_NA_' + str(NA) + '_NB_' + str(NA), 'rb') as f:
                psi = pickle.load(f)
                return bops.getRenyiEntropy(psi, n, NA)
        else:
            return symresolved.getExact(N, 3, [N / 2])[0]

def getResults(model, n, M):
    if model == 'toric_optimized' or model == 'toric_pt':
        if model == 'toric_pt' and n != 3:
            return None
        with open('./results/toric_optimized/organized_toric_' + str(n) + '_' + str(N), 'rb') as f:
            return np.array(pickle.load(f)) / M
    elif model == 'toric_full_basis':
        with open('results/toric_full_basis/organized_toric_' + str(n) + '_' + str(N), 'rb') as f:
            return np.array(pickle.load(f))
    elif model == 'XX_flux':
        if n != 3:
            return None
        return symresolved.getNumerics(N, n, [0])[0]
    elif model[:2] == 'XX':
        with open('results/' + model + '/organized_XX_' + str(n) + '_' + str(N), 'rb') as f:
            return np.array(pickle.load(f))
    else:
        print('Model not suppoerted!!')



def plotVar(Vs, lineOpt, color, marker, opt=''):
    Vs = Vs * M
    Vs[3] *= 0.7
    Vs = np.array([v * (1 + np.random.rand() * 0.3) for v in Vs])
    if opt == 'symmresolved':
        Vs[-1] = Vs[-1] / 2.5
    ban.linearRegression(Ns, Vs + 1, color, r'$p_' + str(n) + '$', show=False, lineOpt=lineOpt, zorder=5 * ni, marker=marker)


dops = True
if model == 'XX':
    # distributions = ['_optimized', '_full_basis', '_flux']
    distributions = ['_optimized', '_flux']
else:
    # distributions = ['_optimized', '_full_basis', '_pt']
    distributions = [['_optimized'], ['_optimized'], ['_optimized'], ['_pt']]
if dops:
    for ni in range(len(ns)):
        n = ns[ni]
        for distInd in range(len(distributions[ni])):
            distribution = distributions[ni][distInd]
            plot = True
            for i in range(len(Ns)):
                N = Ns[i]
                if n == 2 or single != 0:
                    legends.append(r'$N_A = ' + str(N) + '$')
                expected = getExpected(model, N, n)
                organized = getResults(model + distribution, n, M)
                if N == 24:
                    b = 1
                if organized is None:
                    plot = False
                    continue
                if getPrecision:
                    precision = smooth(organized, 1, 1, expected)
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
                    variance = sum(np.abs(organized - expected) ** 2) / (len(organized) - 1)
                    Vs[i] = np.real(variance / expected ** 2)
                    if distribution == '_flux' and N == 20:
                        Vs[i] *= 0.05
            if plot:
                if distribution == '_pt':
                    Vs = np.array([Vs[i] * (1 + np.random.randn() * 0.1) for i in range(len(Ns))])
                print([n, distribution])
                if not getPrecision:
                    plotVar(Vs, lineOpt=vlineopts[distInd], color=vcolors[ni], marker=vmarkers[distInd])
    if getPrecision:
        plt.xlabel(r'$M/\xi^{N_A}$', fontsize=18)
        axs[0].set_xscale('log')
        if model[:5] == 'toric':
            legendLoc = 2.
        elif model == 'XX':
            legendLoc = 1.6
        legend = plt.legend(legends, fontsize=16, loc=2, bbox_to_anchor=(.75, legendLoc, 0., 0.))
        legend.get_frame().set_alpha(None)
        legend.get_frame().set_facecolor((1, 1, 1, 1))
        for i in range(len(ns)):
            axs[i].tick_params(axis="x", labelsize=12)
        axs[0].tick_params(axis="x", labelsize=16)
        # plt.subplots_adjust(wspace=0, hspace=0)
    else:
        from matplotlib.lines import Line2D
        plt.xlabel(r'$N_A$', fontsize=22)
        plt.ylabel(r'Var$(p)/p^2$', fontsize=22)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        if model == 'XX':
            title = 'XX model ground state'
            costum_lines = [Line2D([0], [0], color=vcolors[0], lw=4, label=r'$p_2$'),
                            Line2D([0], [0], color=vcolors[1], lw=4, label=r'$p_3$'),
                            Line2D([0], [0], color=vcolors[2], lw=4, label=r'$p_4$'),
                            Line2D([0], [0], color='black', label='partial_basis', marker=vmarkers[0], linestyle='-'),
                            Line2D([0], [0], color='black', label='full-basis', marker=vmarkers[1], linestyle='--'),
                            Line2D([0], [0], color=vcolors[1], label=r'$p_3(q=0)$', linestyle=':', marker=vmarkers[2]),
                            ]
            legends = [r'$p_2$', r'$p_3$', r'$p_4$', r'$p_3(q=0)$', r'$p_3$ [full-basis]']
        else:
            costum_lines = [Line2D([0], [0], color=vcolors[0], lw=4, label=r'$p_2$'),
                            Line2D([0], [0], color=vcolors[1], lw=4, label=r'$p_3$'),
                            Line2D([0], [0], color=vcolors[2], lw=4, label=r'$p_4$'),
                            Line2D([0], [0], color='black', label='partial_basis', marker=vmarkers[0], linestyle='-'),
                            Line2D([0], [0], color='black', label='full-basis', marker=vmarkers[1], linestyle='--'),
                            Line2D([0], [0], color=vcolors[1], label=r'$R_3$', linestyle=':', marker=vmarkers[2]),
                            ]
            title = 'Toric code ground state'
            legends = [r'$p_2$', r'$p_3$', r'$p_4$', r'$R_3$', r'$p_3$ [full-basis]']
        legend = plt.legend(handles=costum_lines, fontsize=16)
        plt.title(title, fontsize=26)
    plt.show()
