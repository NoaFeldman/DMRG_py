import basicOperations as bops
import numpy as np
import tensornetwork as tn
from scipy import linalg
import math
import test


def getTrotterGates(N, d, onsiteTerms, neighborTerms, dt):
    trotterGates = []
    for i in range(N - 1):
        currTerm = neighborTerms[i] + np.kron(onsiteTerms[i], np.eye(d))
        if i == N - 2:
            currTerm += np.kron(np.eye(d), onsiteTerms[i + 1])
        currTerm = linalg.expm(-0.5 * 1j * dt * currTerm)
        currTerm = np.reshape(currTerm, (d, d, d, d))
        trotterGates.append(tn.Node(currTerm, name=('trotter' + str(i)),
                                    axis_names=['s' + str(i) + '*', 's' + str(i + 1) + '*', 's' + str(i),
                                                's' + str(i + 1)],
                                    backend=None))
    return trotterGates


def trotterSweep(trotterGates, psi, startSite, endSite, maxBondDim=1024):
    psiCopy = bops.copyState(psi)
    truncErr = 0
    N = len(psi)
    for k in [endSite - i for i in range(startSite, endSite)]:
        M = bops.multiContraction(psiCopy[k - 1], psiCopy[k], [2], [0])
        M = bops.permute(bops.multiContraction(M, trotterGates[k - 1], [1, 2], [0, 1]), [0, 2, 3, 1])
        [l, r, currTruncErr] = bops.svdTruncation(M, M[:2], M[2:],
                                                  '<<', maxBondDim, leftName='site' + str(k - 1),
                                                  rightName='site' + str(k), edgeName='v' + str(k))
        if currTruncErr > truncErr:
            truncErr = currTruncErr
        psiCopy[k] = r
        psiCopy[k - 1] = l
        bops.multiContraction(r, r, [1, 2], [1, 2, '*'])
    for k in range(startSite, endSite):
        M = bops.multiContraction(psiCopy[k], psiCopy[k + 1], [2], [0])
        M = bops.permute(bops.multiContraction(M, trotterGates[k], [1, 2], [0, 1]), [0, 2, 3, 1])
        [l, r, currTruncErr] = bops.svdTruncation(M, M[:2], M[2:],
                                                  '>>', maxBondDim, leftName='site' + str(k),
                                                  rightName='site' + str(k + 1), edgeName='v' + str(k + 1))
        if currTruncErr > truncErr:
            truncErr = currTruncErr
        psiCopy[k] = l
        psiCopy[k + 1] = r
    # For imaginary time propagation, renormalize state.
    norm = bops.getOverlap(psiCopy, psiCopy)
    psiCopy[N - 1].tensor = psiCopy[N - 1].tensor / math.sqrt(abs(norm))
    return psiCopy, truncErr
