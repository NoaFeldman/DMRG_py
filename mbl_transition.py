import numpy as np
import matplotlib.pyplot as plt
import pickle

N = 8
repetitions = 1000
Ws = [np.round(0.1 * i, 8) for i in range(100)]

Z = np.array([1, -1])
H_base = np.zeros((2**N, 2**N))
hopping = np.array([[0, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 0]])
for i in range(N - 1):
    H_base += np.kron(np.eye(2**i), np.kron(hopping, np.eye(2**(N - i - 2))))
iprs_avgs = np.zeros((N, len(Ws)))
for interaction_length in range(N):
    print('il = ' + str(interaction_length))
    for wi in range(len(Ws)):
        W = Ws[wi]
        print('wi = ' + str(wi))
        iprs = np.zeros(2**N)
        for r in range(repetitions):
            H = np.copy(H_base)
            for l in range(interaction_length + 1):
                for site in range(N - l):
                    H += np.kron(np.eye(2**site), np.kron(
                                 np.diag([np.random.uniform(-W/2, W/2) for i in range(2**(l+1))]),
                                 np.eye(2**(N - site - l - 1))))
            evals, evecs = np.linalg.eigh(H)
            iprs_curr = np.array([np.sum(np.abs(evecs[:, i]**4)) for i in range(len(evals))])
            iprs += iprs_curr
        iprs /= repetitions
        iprs_avgs[interaction_length, wi] = np.sum(iprs) / len(iprs)
pickle.dump(iprs_avgs, open('results/mbl_stuff_' + str(N), 'wb'))
plt.pcolormesh(list(range(N)), Ws, iprs_avgs)
plt.show()