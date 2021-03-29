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
    if '_m_' in file:
        with open(indir + '/' + file, 'rb') as f:
            organized.append(pickle.load(f))
with open(outdir + '/' + 'organized_' + option + '_' + n + '_' + NA, 'wb') as f:
    pickle.dump(organized, f)
