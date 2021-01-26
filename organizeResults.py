import pickle
import numpy as np
import os
import sys

n = sys.argv[1]
l = sys.argv[2]
option = sys.argv[3]
homedir = sys.argv[4]

dir = homedir + option + n + l
results = []
for file in os.listdir(dir):
    with open(dir + '/' + file, 'rb') as f:
        results.append(pickle.load(f))
with open(homedir + 'organized_' + option + '_' + n + '_' + l, 'wb') as f:
    pickle.dump(results, f)
