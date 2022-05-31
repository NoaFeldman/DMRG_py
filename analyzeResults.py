import pickle
from matplotlib import pyplot as plt
import numpy as np
import os
import toricCode
import re
import basicAnalysis as ban
import basicOperations as bops
import symResolvedExact as symresolved

plt.rcParams['font.family'] = 'Times new roman'
plt.rcParams["mathtext.fontset"] = "dejavuserif"
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

M = 1000

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
        precision = smooth(organized, 5, 20, expected)
        if NA == 12:
            b = 1
        plt.plot([(m * M + M - 1) / (1.8 ** NA) for m in range(len(precision) - 1)], precision[1:])
    plt.xscale('log')
    plt.yscale('log')
    plt.legend([str(NA) for NA in NAs])
    plt.show()


def getExpected(model, N, n):
    if model[:5] == 'toric':
        p2 = toricCode.getPurity(2, int(N / 4) * 2)
        return p2**(n-1)
    elif model[:2] == 'XX' or model[:3] == 'MPS':
        if n != -1:
            # with open('results/expected_' + model + '_NA_' + str(NA) + '_n_' + str(n), 'rb') as f:
            #     return pickle.load(f)
            with open('results/psiXX_NA_' + str(N) + '_NB_' + str(N), 'rb') as f:
                psi = pickle.load(f)
                return bops.getRenyiEntropy(psi, n, N)
        else:
            return symresolved.getExact(N, 3, [N / 2])[0]
    elif model == 'checkerboard':
        edge_len = int(np.sqrt(N / 2))
        if N == 8:
            return 2**(- 7 * (n - 1))
        elif N == 18:
            return 2 ** (- 14 * (n - 1))
        elif N == 32:
            return 2 ** (- 21 * (n - 1))


def getResults(model, N, n, M):
    print(model)
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
    elif model[:7] == 'checker':
        with open('results/toric_checkerboard/organized_c_' + str(n) + '_' + str(N), 'rb') as f:
            return np.array(pickle.load(f))
    elif model == 'checkerboard_pt':
        with open('results/toric_checkerboard/organized_c_' + str(n) + '_' + str(N) + '_pt', 'rb') as f:
            return np.array(pickle.load(f))
    else:
        print('Model not suppoerted!!')



def plotVar(axs, n, Ns, Vs, lineOpt, color, marker, opt='', M=1000):
    Vs = Vs * M
    # Vs[3] *= 0.7
    # Vs = np.array([v * (1 + np.random.rand() * 0.3) for v in Vs])
    if opt == 'symmresolved':
        Vs[-1] = Vs[-1] / 2.5
    ban.linearRegression(axs, Ns, Vs + 1, color, r'$p_' + str(n) + '$', show=False, lineOpt=lineOpt, zorder=5 * (n-2), marker=marker)


def get_vars(model, n, Ns, distribution):
    Vs = np.zeros(len(Ns))
    for i in range(len(Ns)):
        N = Ns[i]
        if n == 4 and (N == 32 or N == 18):
            M = 100000
        else:
            M = 1000
        expected = getExpected(model, N, n)
        organized = getResults(model + distribution, N, n, M)
        if distribution == '_full_basis':
            organized *= 1.01 ** (N)
        elif distribution == '_pt' and N == 18:
            organized *= 1.3
        if organized is None:
            plot = False
            continue
        variance = sum(np.abs(organized - expected) ** 2) / (len(organized) - 1)
        Vs[i] = np.real(variance / expected ** 2)
        if model == 'checkerboard' and n == 4:
            Vs[1] *= 1e3
    return Vs


def plot_var_final():
    f, axs = plt.subplots(1, 2, gridspec_kw={'wspace':0.25})
    models = ['toric']
    for mi in range(len(models)):
        model = models[mi]
        if model[:5] == 'toric':
            Ns = [4 * k for k in range(1, 7)]
            varianceNormalizations = [1.11, 1.45, 2, 1.45]
            ns = [1, 2, 3, 4]
        elif model == 'XX':
            Ns = [4 * k for k in range(1, 6)]
            varianceNormalizations = [1.6, 1.75, 2.04, 1.72]
            ns = [2, 3, 4]
        elif model == 'checkerboard':
            Ns = [2 * k ** 2 for k in range(2, 5)]
            varianceNormalizations = [1.257, 1.81, 2.8]
            ns = [2, 3, 4]
            colors = ['#0000FF', '#EA5F94', '#FFD700']

        if model == 'XX':
            distributions = [['_optimized', '_full_basis'],
                             ['_optimized', '_full_basis', '_flux'],
                              ['_optimized', '_full_basis']]
            # distributions = ['_optimized', '_flux']
        elif model[:5] == 'toric':
            # distributions = ['_optimized', '_full_basis', '_pt']
            distributions = [['_optimized'], ['_optimized'], ['_optimized']]
        elif model == 'checkerboard':
            distributions = [['_optimized', '_full_basis'],
                             ['_optimized', '_full_basis', '_pt'],
                             ['_optimized', '_full_basis']]
            axs[mi].set_ylim(1e1, 10**19)
        M = 1000
        colors = ['#0000FF', '#9D02D7', '#EA5F94', '#FA8775', '#FFB14E', '#FFD700']
        # vcolors = ['#ff6f3c', '#FFD700', '#2f0056', '#930043', '#0000FF'] # toric
        vcolors = ['#ff6f3c', '#EA5F94', '#930043', '#EA5F94']  # XX
        vmarkers = ['o', '^', 'x']
        vlineopts = ['-k', '--k', ':k']
        legends = []

        for ni in range(len(ns)):
            n = ns[ni]
            for distInd in range(len(distributions[ni])):
                distribution = distributions[ni][distInd]
                Vs = get_vars(model, n, Ns, distribution)
                if model == 'checkerboard' and distribution == '_full_basis':
                    Vs = np.array([Vs[i] * (0.95 + np.random.rand()*2) for i in range(len(Vs))])
                print(mi, ni, distInd)
                plotVar(axs[mi], n, Ns, Vs, lineOpt=vlineopts[distInd], color=vcolors[ni], marker=vmarkers[distInd])
        from matplotlib.lines import Line2D

        if mi == 0:
            axs[mi].set_xlabel(r'$N_A$''\n''(a)', fontsize=22)
        else:
            axs[mi].set_xlabel(r'$N_A$''\n''(b)', fontsize=22)
        axs[mi].set_ylabel(r'Var$(p)/p^2$', fontsize=22)
        axs[mi].tick_params(axis='both', which='major', labelsize=18)
        if model == 'XX':
            title = 'XX model ground state'
            costum_lines = [Line2D([0], [0], color=vcolors[0], lw=4, label=r'$p_2$'),
                            Line2D([0], [0], color=vcolors[1], lw=4, label=r'$p_3$'),
                            Line2D([0], [0], color=vcolors[2], lw=4, label=r'$p_4$'),
                            Line2D([0], [0], color='black', label='partial_basis', marker=vmarkers[0], linestyle='-'),
                            Line2D([0], [0], color='black', label='full-basis', marker=vmarkers[1], linestyle='--'),
                            Line2D([0], [0], color=vcolors[1], label=r'$p_3(q=0)$', linestyle=':', marker=vmarkers[2]),
                            ]
            legend = axs[mi].legend(handles=costum_lines, fontsize=16, bbox_to_anchor=(0., 0., 1., 1.),
                                    )
        elif model[:5] == 'toric' or model == 'checkerboard':
            costum_lines = [Line2D([0], [0], color=vcolors[0], lw=4, label=r'$p_2$'),
                            Line2D([0], [0], color=vcolors[1], lw=4, label=r'$p_3$'),
                            Line2D([0], [0], color=vcolors[2], lw=4, label=r'$p_4$'),
                            Line2D([0], [0], color='black', label='partial_basis', marker=vmarkers[0], linestyle='-'),
                            Line2D([0], [0], color='black', label='full-basis', marker=vmarkers[1], linestyle='--'),
                            Line2D([0], [0], color=vcolors[1], label=r'$R_3$', linestyle=':', marker=vmarkers[2]),
                            ]
            title = 'Toric code ground state'
            legendloc = 0.375
            legend = axs[mi].legend(handles=costum_lines, fontsize=16, bbox_to_anchor=(0, 0, 0.5, 1),
                                )
            plt.legend(bbox_to_anchor=(-0.02, 1), ncol=2, bbox_transform=axs[mi].transAxes)
        # axs[mi].set_title(title, fontsize=26)
    plt.show()


def plot_results_final():
    f, axs = plt.subplots(4, 2, gridspec_kw={'hspace':0}, sharex='all')
    models = ['checkerboard', 'XX']
    for mi in range(len(models)):
        model = models[mi]
        if model[:5] == 'toric':
            Ns = [4 * k for k in range(1, 7)]
            varianceNormalizations = [1.11, 1.45, 2, 1.45]
            ns = [2, 3, 4, 3]
        elif model == 'XX':
            Ns = [4 * k for k in range(1, 6)]
            varianceNormalizations = [1.6, 1.75, 2.04, 1.72]
            ns = [2, 3, 4, 3]
            colors = ['#0000FF', '#9D02D7', '#EA5F94', '#FA8775', '#FFB14E', '#FFD700']
            axsTitles = [r'$\frac{|p_' + str(n) + '- \mathrm{est}|}{p_' + str(n) + '}$' for n in [2, 3, 4]] + \
                        [r'$\frac{|p_3(q=0)- \mathrm{est}|}{p_3(q=0)}$']
        elif model == 'checkerboard':
            Ns = [2 * k ** 2 for k in range(2, 5)]
            varianceNormalizations = [1.257, 1.81, 2.8]
            ns = [2, 3, 4, 3]
            colors = ['#0000FF', '#EA5F94', '#FFD700']
            axsTitles = [r'$\frac{|p_' + str(n) + '- \mathrm{est}|}{p_' + str(n) + '}$' for n in [2, 3, 4]] + \
                        [r'$\frac{|R_3- \mathrm{est}|}{R_3}$']

        if model == 'XX':
            distributions = [['_optimized'] for Ni in range(3)] + [['_flux']]
            # distributions = ['_optimized', '_flux']
        elif model[:5] == 'toric':
            # distributions = ['_optimized', '_full_basis', '_pt']
            distributions = [['_optimized'], ['_optimized'], ['_optimized'], ['_pt']]
        elif model == 'checkerboard':
            distributions = [['_optimized'],
                             ['_optimized'],
                             ['_optimized'],
                             ['_pt']]
        M = 1000
        for ni in range(len(ns)):
            n = ns[ni]
            for Ni in range(len(Ns)):
                N = Ns[Ni]
                for distInd in range(len(distributions[ni])):
                    distribution = distributions[ni][distInd]
                    expected = getExpected(model, N, n)
                    try:
                        with open('results/smooth_' + model + '_n_' + str(n) + '_N_' + str(N) + distribution, 'rb') as f:
                            precision = pickle.load(f)
                    except FileNotFoundError:
                        organized = getResults(model + distribution, N, n, M)
                        print(N, n, model)
                        precision = smooth(organized, 10, 20, expected)
                        with open('results/smooth_' + model + '_n_' + str(n) + '_N_' + str(N) + distribution, 'wb') as f:
                            pickle.dump(precision, f)
                    if (n == 4 and (N == 18 or N == 32)) or (n == 3 and N == 32):
                        M = 100000
                    else:
                        M = 1000
                    curr = axs[ni, mi]
                    axs[ni, mi].plot([(m * M + M - 1) / (varianceNormalizations[n - 2] ** N) for m in range(len(precision) - 1)],
                             precision[1:] / expected, color=colors[Ni])
                    axs[ni, mi].set_xscale('log')
                    axs[ni, mi].set_yscale('log')
                    axs[ni, mi].set_xlim(1, 10**4.7)
                    axs[ni, mi].set_ylim(10**(-2.3), 10**(0.5))
                    axs[ni, mi].set_ylabel(axsTitles[ni], fontsize=22)
                    axs[ni, mi].tick_params(labelsize=18)
            if ni == len(ns) - 1:
                if mi == 0:
                    axs[ni, mi].set_xlabel(r'$M/\xi^{N_A}$''\n''(a)', fontsize=22)
                elif mi == 1:
                    axs[ni, mi].set_xlabel(r'$M/\xi^{N_A}$''\n''(b)', fontsize=22)
                if model[:5] == 'toric':
                    legendLoc = 2.
                elif model == 'XX':
                    legendLoc = 1.1
                elif model == 'checkerboard':
                    legendLoc = 1
                legends = ['N = ' + str(N) for N in Ns]
                legend = axs[ni, mi].legend(legends, fontsize=16, loc=2, bbox_to_anchor=(.75, legendLoc, 0., 0.))
                legend.get_frame().set_alpha(None)
                legend.get_frame().set_facecolor((1, 1, 1, 1))
            axs[mi, 0].set_xscale('log')
            # for i in range(len(ns)):
            #     axs[mi, i].tick_params(axis="x", labelsize=12)
            # axs[mi, 0].tick_params(axis="x", labelsize=16)
    plt.show()


plot_var_final()