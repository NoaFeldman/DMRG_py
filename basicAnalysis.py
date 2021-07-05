import numpy as np
import matplotlib.pyplot as plt

# Linear regression, based on https://stackoverflow.com/questions/6148207/linear-regression-with-matplotlib-numpy
def linearRegression(Ns, Vs, color='blue', label='', show=True, lineOpt='-k', zorder=0):
    coef = np.polyfit(Ns, np.log2(Vs), 1)
    print(coef)
    poly1d_fn = np.poly1d(coef)
    plt.plot(Ns, 2 ** poly1d_fn(Ns), lineOpt, color=color, zorder=zorder)
    plt.scatter(Ns, Vs, color=color, label=label, zorder=zorder)
    plt.yscale('log')
    plt.xticks(Ns)
    if show:
        plt.show()
    return coef[0]

def variance(organized, expected=None):
    organized = np.array(organized)
    if expected is None:
        expected = np.average(organized)
    variance = np.sum(np.abs(organized - expected) ** 2) / (len(organized) - 1)
    return variance
