import pickle
from matplotlib import pyplot as plt
import numpy as np


ls = [6, 8, 10]
M = 30
for l in ls:
    with open('localMC_l_' + str(l) + '_M_' + str(M) + '_l_' + str(l) + '_M_' + str(M), 'rb') as f:
        mc = pickle.load(f)
    with open('exact_l_' + str(l), 'rb') as f:
        exact = pickle.load(f)
    plt.plot([(i + M-1) / l**2 for i in range(l**2)], np.abs(mc - exact) / exact)
plt.show()