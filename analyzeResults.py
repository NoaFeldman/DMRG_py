import pickle
from matplotlib import pyplot as plt
import numpy as np
import os
import toricCode
import re

d = 2


# Linear regression, based on https://stackoverflow.com/questions/6148207/linear-regression-with-matplotlib-numpy
def linearRegression(Ns, Vs):
    coef = np.polyfit(Ns, np.log2(Vs), 1)
    print(coef)
    poly1d_fn = np.poly1d(coef)
    plt.plot(Ns, Vs, 'yo', Ns, 2**poly1d_fn(Ns), '--k')
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
Ns = [4, 8, 12]
legends = []
option = 'experimental'
Vs = np.zeros(len(Ns))
ns = [2, 3, 4]
ns = [4]
varianceNormalizations = [1.23, 1.51, 2.01]
p2s = []
for i in range(len(Ns)):
    p2s.append(toricCode.getPurity(i + 1))
dops = True
if dops:
    for n in ns:
        precisions = []
        for i in range(len(Ns)):
            N = Ns[i]
            spaceSize = d**N
            legends.append('N = ' + str(N))
            organized = []
            # for j in range(100):
            #     dirname = rootdir + '/' + option + str(n) + str(i+1)
            #     if j > 0:
            #         dirname += '_' + str(j)
            #     if os.path.exists(dirname):
            #         for filename in os.listdir(dirname):
            #             with open(dirname + '/' + filename, 'rb') as f:
            #                 curr = pickle.load(f)
            #                 organized.append(curr)

            with open('./results/' + str(findResults(n, N, 'experimental')), 'rb') as f:
                organized = np.array(pickle.load(f))
            organized = organized[organized < 50]
            # with open('./results/' + 'organized_p' + str(n) + '_N_' + str(N) + '_' + str(len(organized)), 'wb') as f:
            #     pickle.dump(organized, f)
            p2 = p2s[i]
            expected = p2**(n-1)
            numOfExperiments = 1
            numOfMixes = 2
            precision = np.zeros(int(len(organized) / numOfExperiments))
            for mix in range(numOfMixes):
                np.random.shuffle(organized)
                for j in range(1, int(len(organized)/numOfExperiments)):
                    currPrecision= np.average([np.abs(np.average( \
                        organized[c * int(len(organized)/numOfExperiments):c * int(len(organized)/numOfExperiments)+j]) - expected) \
                                               for c in range(numOfExperiments)])
                    # currPrecision= np.average([ \
                    #     (np.average(organized[c * int(len(organized)/numOfExperiments):c * int(len(organized)/numOfExperiments)+j]) \
                    #      - expected)**2 \
                    #     for c in range(numOfExperiments)])
                    if mix == 0:
                        precision[j] = currPrecision
                    else:
                        precision[j] = (precision[j] * mix + currPrecision) / (mix + 1)
            plt.plot([(m * M + M - 1) / (varianceNormalizations[n - 2] ** N) for m in range(len(precision)-1)],
                     precision[1:] / expected)
            # plt.plot([(m * M + M - 1) / (1.52 ** N) for m in range(len(precision)-1)],
            #          precision[1:] / expected)
            variance = np.real(np.average((np.array(organized) - expected)**2 * M))
            Vs[i] = np.real(variance / expected**2)
        plt.xlabel(r'$M/(' + str(varianceNormalizations[n-2]) + '^N)$')
        # plt.xlabel(r'$M/(' + str(1.53) + '^N)$')
        plt.ylabel(r'|$p_' + str(n) + '$ - est|/$p_' + str(n) + '$')
        # plt.ylabel(r'|$R_' + str(n) + '$ - est|/$R_' + str(n) + '$')
        plt.legend(legends)
        plt.show()
        linearRegression(Ns, Vs)
        plt.xlabel(r'$N_A$')
        plt.ylabel(r'Var$(p_' + str(n) + ')/p^2_' + str(n) + '$')
        plt.show()

doR3 = False
if doR3:
    for i in range(len(Ns)):
        N = Ns[i]
        m = M - 1
        estimation = []
        organized = []
        legends.append('N = ' + str(N))
        while os.path.isfile('./results/complex3' + str(i+1) + '/neg_n_3_N_' + str(N) + '_' + option +
                             '_M_' + str(M) + '_m_' + str(m)):
            with open('./results/complex3' + str(i+1) + '/neg_n_3_N_' + str(N) + '_' + option +
                             '_M_' + str(M) + '_m_' + str(m), 'rb') as f:
                curr = pickle.load(f)
                organized.append(curr)
                if len(estimation) == 0:
                    estimation.append(curr)
                else:
                    estimation.append(np.average(organized))
            m += M
        with open('./results/neg_r3_N_' + str(N) + '_' + str(len(organized)), 'wb') as f:
            pickle.dump(organized, f)
        p2 = toricCode.getPurity(i + 1)
        expected = p2 ** 2
        plt.plot([(m * M + M - 1) / (2 ** N * expected) for m in range(len(estimation))],
                 np.abs(np.array(estimation) - expected) / expected)
        variance = np.average((np.array(organized) - expected) ** 2 * M)
        Vs[i] = variance / expected ** 2
    plt.xlabel(r'$M/(2^N R_3)$')
    plt.ylabel(r'|$R_3$ - est|/$R_3$')
    plt.legend(legends)
    plt.show()
    linearRegression(Ns, Vs)
    plt.xlabel(r'$N_A$')
    plt.ylabel(r'Var$(R_3)/R^2_3$')
    plt.show()
dop3 = False
if dop3:
    for i in range(len(Ns)):
        N = Ns[i]
        spaceSize = d**N
        m = M - 1
        estimation = []
        organized = []
        legends.append('N = ' + str(N))
        while os.path.isfile('./results/renyis/toric_local_vecs_n_3_N_' + str(N) + '_' + option +
                             '_M_' + str(M) + '_m_' + str(m)):
            with open('./results/renyis/toric_local_vecs_n_3_N_' + str(N) + '_' + option +
                             '_M_' + str(M) + '_m_' + str(m), 'rb') as f:
                curr = pickle.load(f)
                organized.append(curr)
                if len(estimation) == 0:
                    estimation.append(curr)
                else:
                    estimation.append((estimation[-1] * len(estimation) + curr) / (len(estimation) + 1))
            m += M
        with open('./results/organized_p3_N_' + str(N) + '_' + str(len(organized)), 'wb') as f:
            pickle.dump(organized, f)
        p2 = toricCode.getPurity(i + 1)
        expected = p2 ** 2
        plt.plot([(m * M + M - 1) / (2 ** N) for m in range(len(estimation))],
                 np.abs(np.array(estimation) - expected) / expected)
        variance = np.average((np.array(organized) - expected) ** 2 * M)
        Vs[i] = variance / expected ** 2
    plt.xlabel(r'$M/(2^N))$')
    plt.ylabel(r'|$p_3$ - est|/$p_3$')
    plt.legend(legends)
    plt.show()
    linearRegression(Ns, Vs)
    plt.xlabel(r'$N_A$')
    plt.ylabel(r'Var$(p_3)/p^2_3$')
    plt.show()


dop4 = False
if dop4:
    for i in range(len(Ns)):
        N = Ns[i]
        spaceSize = d**N
        m = M - 1
        estimation = []
        organized = []
        p2 = toricCode.getPurity(i + 1)
        expected = p2 ** 3
        legends.append('N = ' + str(N))
        while os.path.isfile('./results/complex4' + str(i+1) + '/toric_local_vecs_n_4_N_' + str(N) + '_' + option +
                             '_M_' + str(M) + '_m_' + str(m)):
            with open('./results/complex4' + str(i+1) + '/toric_local_vecs_n_4_N_' + str(N) + '_' + option +
                             '_M_' + str(M) + '_m_' + str(m), 'rb') as f:
                curr = pickle.load(f)
                organized.append(curr)
                if len(estimation) == 0:
                    estimation.append(curr)
                else:
                    estimation.append((estimation[-1] * len(estimation) + curr) / (len(estimation) + 1))
            m += M
        with open('./results/organized_p4_N_' + str(N) + '_' + str(len(organized)), 'wb') as f:
            pickle.dump(organized, f)
        plt.plot([(m * M + M - 1) / (2**N) for m in range(len(estimation))],
                 np.abs(np.array(estimation) - expected) / expected)
        variance = np.average((np.array(organized) - expected) ** 2 * M)
        Vs[i] = variance / expected ** 2
    plt.xlabel(r'$M/(2^N))$')
    plt.ylabel(r'|$p_4$ - est|/$p_4$')
    plt.legend(legends)
    plt.show()
    linearRegression(Ns, Vs)
    plt.xlabel(r'$N_A$')
    plt.ylabel(r'Var$(p_4)/p^2_4$')
    plt.show()
