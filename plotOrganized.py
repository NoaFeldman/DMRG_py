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
n = 2
Ns = [4, 8, 12, 16, 20, 24]
Vs = np.zeros(len(Ns))
for i in range(len(Ns)):
    N = Ns[i]
    with open('results/expected_' + option + '_NA_' + str(N) + '_n_' + str(n), 'rb') as f:
        expected = pickle.load(f)
    with open('results/organized_' + option + '_' + str(n) + '_' + str(N), 'rb') as f:
        organized = pickle.load(f)
    with open('results/conserved_' + option + '_' + str(n) + '_' + str(N), 'rb') as f:
        converged = pickle.load(f)
    Vs[i] = np.sum(np.abs(organized - expected)**2) / expected**2
linearRegression(Ns[1:], Vs[1:], 'blueviolet', r'$p_2$')
plt.scatter(Ns, Vs)
plt.yscale('log')
plt.show()
