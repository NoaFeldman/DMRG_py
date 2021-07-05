import pickle
import numpy as np
from matplotlib import pyplot as plt
import toricCode
import basicAnalysis as ban

Ns = [4*k for k in range(1, 6)]
n = 1
def compareXYZ():
    options = ['xy', 'xz', 'yz']
    colors = ['blue', 'green', 'orange']
    for i in range(len(options)):
        option = options[i]
        vs = np.zeros(len(Ns))
        for j in range(len(Ns)):
            N = Ns[j]
            with open(dir + 'organized_toric_' + option + '_' + str(n) + '_' + str(N), 'rb') as f:
                organized = np.array(pickle.load(f)) / 1000
            vs[j] = ban.variance(organized) * 1000
        ban.linearRegression(Ns, vs+1, color=colors[i], show=False)
    plt.legend(options)
    plt.show()


def colorMap(thetas, phis, dir, NA):
    w = 2
    h = 6
    n = 1

    results = np.zeros((len(thetas), len(phis)))
    for i in range(len(thetas)):
        theta = np.round(thetas[i], 1)
        for j in range(len(phis)):
            phi = np.round(phis[j], 1)
            with open(dir + 't_' + str(theta) + '_p_' + str(phi) + '_' + str(n) + '_' + str(NA), 'rb') as f:
                organized = np.array(pickle.load(f)) / 1000
            vars = np.zeros(len(organized))
            # expected = (toricCode.getPurity(w, h))**(n-1)
            expected = 1
            showConvergence = False
            if showConvergence:
                for k in range(2, len(organized)):
                    avg = np.average(organized[:k])
                    var = np.sum(np.abs(organized[:k] - avg)**2 / (k - 1)) / expected**2
                    vars[k] = var
                plt.plot(vars)
                plt.title(r'$\theta = ' + str(theta) + ', \phi = ' + str(phi) + '$')
                plt.show()
            var = np.sum(np.abs(organized - expected) ** 2 / (len(organized) - 1)) / expected ** 2
            results[i, j] = var * np.sqrt(1000)
            if i == 0 and j in [2, 3]:
                print(var)
    plt.pcolormesh(thetas, phis, results)
    plt.colorbar()
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$\phi$')
    plt.show()
# colorMap([0.1 * k for k in range(11)], [0.1 * k for k in range(11)], 'results/XXPhases/organized_XX_', 20)
ts = [np.round(0.1 * k, 1) for k in range(6)]
ps = [np.round(0.1 * k, 1) for k in range(6)]
res = np.zeros((len(ts), len(ps)))
for t in range(len(ts)):
    for p in range(len(ps)):
        Ns = [4 * k for k in range(1, 6)]
        vs = np.zeros(len(Ns))
        for n in range(len(Ns)):
            with open('results/XXVars/XXVar_NA_' + str(Ns[n]) + '_t_' + str(np.round(ts[t], 1)) + '_p_' + str(np.round(ps[p], 1)), 'rb') as f:
                vs[n] = pickle.load(f)[1]
        V = ban.linearRegression(Ns, vs, show=False)
        res[t, p] = V
plt.show()
plt.pcolormesh(ts, ps, res)
plt.colorbar()
plt.xlabel(r'$\theta/\pi$')
plt.ylabel(r'$\phi/\pi$')
plt.show()
