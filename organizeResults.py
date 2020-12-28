import pickle
import numpy as np
import os.path
import sys

d = 2

M = 1000
N = 4 * int(sys.argv[1])
n = int(sys.argv[2])
option = 'complex'
dirs = []
for i in range(3, len(sys.argv) - 1):
    dirs.append(sys.argv[i])
outdir = sys.argv[len(sys.argv) - 1]
organized = []
for dir in dirs:
    m = M - 1
    while os.path.isfile(dir + '/toric_local_vecs_N_' + str(N) + '_' + option +
                         '_M_' + str(M) + '_m_' + str(m)):
        with open(dir + '/toric_local_vecs_N_' + str(N) + '_' + option +
                         '_M_' + str(M) + '_m_' + str(m), 'rb') as f:
            organized.append(pickle.load(f))
        m += M
    while os.path.isfile(dir + '/toric_local_vecs_n_'+ str(n) + '_N_' + str(N) +'_' + option + '_M_' +
                         str(M) + '_m_' + str(m)):
        with open(dir + '/toric_local_vecs_n_'+ str(n) + '_N_' + str(N) +'_' + option + '_M_' +
                         str(M) + '_m_' + str(m), 'rb') as f:
            organized.append(pickle.load(f))
        m += M
with open(outdir + '/' + option + '_n_'+ str(n) + '_N_' + str(N) + '_' + str(len(organized)), 'wb') as f:
    pickle.dump(organized, f)
