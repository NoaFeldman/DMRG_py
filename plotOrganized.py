import pickle
from matplotlib import pyplot as plt
import numpy as np
import basicAnalysis as ban

d = 2

option = 'MPS'
Ns = [4, 8, 12, 16, 20]
Vs = np.zeros(len(Ns))
scalings = [0.77, 1.01, 1.1]
precisions = []
colors = ['blueviolet', 'blue', 'deepskyblue', 'green', 'yellowgreen', 'orange']
vcolors = ['blueviolet', 'deepskyblue', 'green', 'orange']
f, axs = plt.subplots(4, 1, gridspec_kw = {'wspace':0, 'hspace':0})

for n in [2, 3, 4]:
    for i in range(len(Ns)):
        N = Ns[i]
        if n == 1:
            expected = 1
        else:
            with open('results/expected_' + option + '_NA_' + str(N) + '_n_' + str(n), 'rb') as f:
                expected = pickle.load(f)
        # with open('results/organized_' + model + '_NB_' + str(N) + '_' + str(n) + '_' + str(N), 'rb') as f:
        with open('results/organized_' + option + '_optimized_' + str(n) + '_' + str(N), 'rb') as f:
            organized = np.array(pickle.load(f))
            organized = organized[organized < 40000]
        with open('results/conserved_' + option + '_optimized_' + str(n) + '_' + str(N), 'rb') as f:
            converged = np.array(pickle.load(f))
        avg = np.average(organized)
        var = np.sum(np.abs(organized - np.average(organized)) ** 2) / (len(organized) - 1)
        Vs[i] = var / expected**2 * 1000
        print(N, len(organized), np.round(avg/expected, 3), np.round(Vs[i], 3))
        numOfExperiments = 10
        numOfMixes = 1 # 20
        precision = np.zeros(int(len(organized) / numOfExperiments))
        for mix in range(numOfMixes):
            np.random.shuffle(organized)
            for j in range(1, int(len(organized)/numOfExperiments)):
                currPrecision= np.average([np.abs(np.average( \
                    organized[c * int(len(organized)/numOfExperiments):c * int(len(organized)/numOfExperiments)+j]) - expected) \
                                           for c in range(numOfExperiments)])
                if mix == 0:
                    precision[j] = currPrecision
                else:
                    precision[j] = (precision[j] * mix + currPrecision) / (mix + 1)
        precision = precision[1:]

        axs[n-1].plot((np.array(range(len(precision))) + 1) * 1000 / (2**scalings[n-2])**N, np.abs(precision), color=colors[i])
        axs[n-1].set_xscale('log')
        axs[n-1].set_yscale('log')
        axs[n-1].set_ylabel(r'$\frac{p_' + str(n) + '- \text{est}}{p_' + str(n) + '}$', fontsize=16)
    ban.linearRegression(Ns, Vs, show=False)
legend = plt.legend([r'$N_A$ = ' + str(N) for N in Ns], loc=0,
           bbox_to_anchor=(0.25, 4), edgecolor='black')
legend.get_frame().set_alpha(None)
legend.get_frame().set_facecolor((1, 1, 1, 1))
plt.xlabel(r'$M/V^{N_A}$', fontsize=16)
plt.subplots_adjust(wspace=0, hspace=0)
plt.show()