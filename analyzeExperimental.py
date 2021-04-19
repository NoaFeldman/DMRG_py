import socket, pickle
from matplotlib import pyplot as plt
import numpy as np
import os
import toricCode

# Linear regression, based on https://stackoverflow.com/questions/6148207/linear-regression-with-matplotlib-numpy
def linearRegression(Ns, Vs):
    coef = np.polyfit(Ns, np.log2(Vs), 1)
    print(coef)
    poly1d_fn = np.poly1d(coef)
    plt.plot(Ns, Vs, 'yo', Ns, 2**poly1d_fn(Ns), '--k')
    plt.yscale('log')
    plt.xticks(Ns)

NA = 6
option = 'XX'
n = 4
dirname = 'results/experimental/' + option + '_NA_' + str(NA) + '_n_' + str(n)
organized = np.array([])
for filename in os.listdir(dirname):
    with open(dirname + '/' + filename, 'rb') as f:
        curr = pickle.load(f)
        curr = curr[curr!=0]
        organized = np.concatenate([organized, curr[1:]])
    expected = curr[0]
print(expected)
print(np.average(organized))
print(curr[0] / np.average(organized))
with open(dirname + '_organized', 'wb') as f:
    organized = np.concatenate([np.array([expected]), organized])
    pickle.dump(organized, f)


