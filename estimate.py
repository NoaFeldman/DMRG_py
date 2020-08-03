from scipy import linalg
import numpy as np
import basicOperations as bops
import randomMeasurements as rm
import sys
import pickle
import aklt

pkl_file = open('A1Estimations.pkl', 'rb')
A1Estimations = pickle.load(pkl_file)
pkl_file = open('A1Estimations.pkl', 'rb')
A2Estimations = pickle.load(pkl_file)


UtrBase = linalg.expm(1j * np.pi * aklt.sY)


M = 10 # sys.argv[1]
m = 0 # sys.argv[2]
result = 0
for j in range(M):
    curr = 1
    for i in range(aklt.lenA1):
        curr *= np.trace(np.matmul(A1Estimations[m, i, :, :], A1Estimations[j, i, :, :]))
    result += curr
output = open('purity_' + str(m) + '.pkl', 'wb')
pickle.dump(result / M, output)
output.close()

