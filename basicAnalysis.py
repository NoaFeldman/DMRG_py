import numpy as np
import matplotlib.pyplot as plt

# Linear regression, based on https://stackoverflow.com/questions/6148207/linear-regression-with-matplotlib-numpy
def linearRegression(Ns, Vs, color='blue', label='', show=True):
    coef = np.polyfit(Ns, np.log2(Vs), 1)
    print(coef)
    poly1d_fn = np.poly1d(coef)
    # plt.plot(Ns, Vs, 'yo', Ns, 2**poly1d_fn(Ns), '--k', color=color, label='p2')
    plt.scatter(Ns, Vs, color=color, label=label)
    plt.plot(Ns, 2 ** poly1d_fn(Ns), '--k', color=color)
    plt.yscale('log')
    plt.xticks(Ns)
    if show:
        plt.show()