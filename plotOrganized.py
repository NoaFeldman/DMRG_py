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
for n in [1, 2, 3, 4]:
    for i in range(len(Ns)):
        N = Ns[i]
        if n == 1:
            expected = 1
        else:
            with open('results/expected_' + option + '_NA_' + str(N) + '_n_' + str(n), 'rb') as f:
                expected = pickle.load(f)
        with open('results/organized_' + option + '_NB_' + str(N) + '_' + str(n) + '_' + str(N), 'rb') as f:
            organized = np.array(pickle.load(f))
            organized = organized[organized < 40000]
        with open('results/conserved_' + option + '_NB_' + str(N) + '_' + str(n) + '_' + str(N), 'rb') as f:
            converged = np.array(pickle.load(f))
        # Vs[i] = np.sum(np.abs(organized - expected)**2) / expected**2
        # plt.scatter(np.array(range(len(organized))), organized)
        avg = np.average(organized)
        var = np.sum(np.abs(organized - np.average(organized)) ** 2) / (len(organized) - 1)
        Vs[i] = var / expected**2 * 1000
        print(N, len(organized), avg/expected, expected, avg, np.sqrt(var / (len(organized))))
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
        # precision = precision[1:]
        # plt.plot((np.array(range(len(precision))) + 1)  * 1000 / (2**scalings[n-2])**N, np.abs(precision), color=colors[i])
        # precisions.append(precision)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend([r'$N_A$ = ' + str(N) for N in Ns], fontsize=14)
    plt.xlabel(r'$M/V^{N_A}$', fontsize=16)
    plt.ylabel(r'Var$(p)/p^2$', fontsize=16)
    plt.title(r'$p_' + str(n) + '$')
    plt.savefig('/home/noa/Documents/randomPeps/XX_p' + str(n) + '.png')
    plt.clf()
    varCoef = ban.linearRegression(Ns, Vs + 1, 'blueviolet', r'$p_2$')
    # plt.scatter(Ns, Vs)
    # plt.yscale('log')
    # plt.xlabel(r'$N_A$', fontsize=16)
    # plt.ylabel(r'Var$(p)/p^2$', fontsize=16)
    # plt.title(r'$p_' + str(n) + '$')
    # plt.text(x=5, y=10, s=(r'$' + str(np.round(varCoef[0], 2)) + 'x + ' + str(np.round(varCoef[1], 2)) + '$'))
    # plt.savefig('/home/noa/Documents/randomPeps/XX_var_p' + str(n) + '.png')
    # plt.clf()

