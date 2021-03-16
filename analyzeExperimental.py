import pickle
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

Ns = [4, 6]
Vs = [0, 0]
dir = 'results/experimental'
for i in range(len(Ns)):
    res = np.array([])
    NA = Ns[i]
    for filename in os.listdir(dir):
        if 'NA_' + str(NA) in filename:
            with open(dir + '/' + filename, 'rb') as f:
                curr = pickle.load(f)
                res = np.concatenate([res, curr[1:]])
    with open('results/experimental/organized_NA_' + str(NA), 'wb') as f:
        pickle.dump(res, f)
    con = np.zeros(len(res))
    Vs[i] = np.average(res**2 - np.average(res)**2) / np.average(res)**2
    print(curr[0] / np.average(res))
linearRegression(Ns, Vs)
plt.show()
# for i in range(1, len(res)):
#     con[i] = np.average(res[:i])
# plt.plot(con)
# plt.show()
