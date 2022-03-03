import pickle
import numpy as np
import os
import matplotlib.pyplot as plt

opt = 'PEPS'
server = 'astro'

def plot_for_N(N):
    cpun_vs_time = {}
    for file in os.listdir('server_tests/' + server):
        fl = file.split('_')
        if fl[3] == opt and int(fl[5]) == N:
            with open('server_tests/' + server + '/' + file, 'rb') as f:
                t = pickle.load(f)
                cpun_vs_time[int(fl[2])] = t
    print(cpun_vs_time)
    cpuns = cpun_vs_time.keys()
    ts = cpun_vs_time.values()
    ts = [x for _, x in sorted(zip(cpuns, ts), key=lambda pair: pair[0])]
    cpuns = sorted(cpuns)
    with open('server_tests/' + server + '_' + opt + '_N_' + str(N), 'wb') as f:
        pickle.dump([cpuns, ts], f)
    plt.plot(cpuns, ts)

legends = []
for N in [4, 8, 12, 16, 20, 24, 28, 32, 36, 40]:
    plot_for_N(N)
    legends.append(str(N))
plt.legend(legends)
plt.show()