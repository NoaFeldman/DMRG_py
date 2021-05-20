import pickle
from matplotlib import pyplot as plt
import numpy as np

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

option = 'MPS'
n = 3
Ns = [4, 8, 12, 16, 20, 24]
Vs = np.zeros(len(Ns))
precisions = []
for i in range(len(Ns)):
    N = Ns[i]
    with open('results/expected_' + option + '_NA_' + str(N) + '_n_' + str(n), 'rb') as f:
        expected = pickle.load(f)
    with open('results/organized_' + option + '_' + str(n) + '_' + str(N), 'rb') as f:
        organized = np.array(pickle.load(f))
    with open('results/conserved_' + option + '_' + str(n) + '_' + str(N), 'rb') as f:
        converged = np.array(pickle.load(f))
    # Vs[i] = np.sum(np.abs(organized - expected)**2) / expected**2
    # plt.scatter(np.array(range(len(organized))), organized)
    avg = np.average(organized)
    var = np.sum(np.abs(organized - expected) ** 2) / (len(organized) - 1)
    Vs[i] = var / expected**2
    print(N, len(organized), avg/expected, expected, avg, np.sqrt(var / (len(organized))))
    numOfExperiments = 10
    numOfMixes = 20
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
    plt.plot(np.array(range(len(precision))) / (2**1.15)**N, np.abs(precision))
    precisions.append(precision)
plt.show()
linearRegression(Ns, Vs, 'blueviolet', r'$p_2$')
plt.scatter(Ns, Vs)
plt.yscale('log')
# plt.show()
