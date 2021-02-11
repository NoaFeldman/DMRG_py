import tensornetwork as tn
import numpy as np
import basicOperations as bops
import DMRG as dmrg
from typing import List

d = 2
randomPossibilities = 2

# vs[i][j] = rabdom vector for copy i, site j
def singleMeasurement(psi: List[tn.Node], vs: List[List[np.array]], choices: List[List[bool]]):
    n = len(vs)
    NA = len(vs[0])
    result = 1
    for copy in range(n):
        psiCopy = bops.copyState(psi)
        for alpha in range(NA - 1, -1, -1):
            overlap = np.matmul(vs[copy][alpha], vs[np.mod(copy+1, n)][alpha])
            case1 = np.round(overlap, 8) != 0
            if case1:
                projectionResult = makeMeasurement(psiCopy, alpha, vs[copy][alpha])
                if projectionResult == 0:
                    return 0
                else:
                    result *= overlap
            else:
                hermitianComponent = choices[copy][alpha] # np.random.randint(2)
                if hermitianComponent:
                    y = 1
                else:
                    y = 1j
                plusVec = vs[copy][alpha] + y * vs[np.mod(copy+1, n)][alpha]
                projectionResult = makeMeasurement(psiCopy, alpha, plusVec)
                if projectionResult == 1:
                    result *= np.matmul(np.conj(np.transpose(plusVec)), vs[np.mod(copy+1, n)][alpha]) / 2
                else:
                    minusVec = vs[copy][alpha] - y * vs[np.mod(copy + 1, n)][alpha]
                    result *= np.matmul(np.conj(np.transpose(minusVec)), vs[np.mod(copy + 1, n)][alpha]) / 2
            psiCopy = bops.shiftWorkingSite(psiCopy, alpha, '<<')
        bops.removeState(psiCopy)
    return result

# Assuming the working site of psi is already site
def makeMeasurement(psi, site, projectedVec):
    toProject = np.kron(projectedVec, np.conj(np.reshape(projectedVec, [2, 1])))
    toProject = toProject / np.trace(toProject)
    localDM = bops.multiContraction(psi[site], psi[site], '02', '02*').tensor
    projectionProbability = np.trace(np.matmul(localDM, toProject)) / np.trace(localDM)
    # Project to the measured vector
    if np.random.uniform(0, 1) < projectionProbability:
        psi[site] = bops.permute(bops.multiContraction(
            psi[site], tn.Node(toProject), '1', '0', cleanOr1=True, cleanOr2=True), [0, 2, 1])
        res = 1
    else:
        psi[site] = bops.permute(bops.multiContraction(
            psi[site], tn.Node(np.eye(d) - toProject), '1', '0', cleanOr1=True, cleanOr2=True), [0, 2, 1])
        res = 0
    return res


def getMeasuredVector(psi, toMeasure, site):
    localDM = bops.multiContraction(psi[site], psi[site], '02', '02*').tensor
