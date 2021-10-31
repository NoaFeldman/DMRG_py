import pickle
import numpy as np
import os
import sys

n = sys.argv[1]
NA = sys.argv[2]
option = sys.argv[3]
indir = sys.argv[4]
outdir = sys.argv[5]

organized = []
for file in os.listdir(indir):
    with open(indir + '/' + file, 'rb') as f:
        try:
            organized.append(pickle.load(f))
        except EOFError:
            pass
with open(outdir + '/' + 'organized_' + option + '_' + n + '_' + NA, 'wb') as f:
    pickle.dump(organized, f)
