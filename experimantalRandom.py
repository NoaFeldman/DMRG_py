import tensornetwork as tn
import numpy as np
import basicOperations as bops
import DMRG as dmrg
from typing import List

d = 2
randomPossibilities = 2

# vs[i][j] = rabdom vector for copy i, site j
def singleMeasurement(psi: List[tn.Node], vs: List[List[np.array]]):
    n = len(vs)
    NA = len(vs[0])
    result = 1
    for copy in range(n):
        psiCopy = bops.copyState(psi)
        for alpha in range(NA - 1, -1, -1):
            overlap = np.matmul(vs[copy][alpha], vs[np.mod(copy+1, n)][alpha]) / 2
            toEstimate = np.kron(vs[copy][alpha], np.conj(np.reshape(vs[np.mod(copy+1, n)][alpha], [2, 1])))
            case1 = np.round(overlap, 8) != 0
            if case1:
                toMeasure = toEstimate / np.trace(toEstimate)
                contribution = applyProjector(psiCopy, toMeasure, alpha)
                if contribution == 0:
                    return 0
                result *= contribution * overlap * 2
            else:
                hermitianComponent = (np.random.randint(2) == 0)
                if hermitianComponent:
                    y = 1
                else:
                    y = 1j
                plusVector = (vs[copy][alpha] + y * vs[np.mod(copy + 1, n)][alpha]) / 2
                projector = np.kron(plusVector, np.conj(np.reshape(plusVector, [2, 1])))
                projectionResult = applyProjector(psiCopy, projector, alpha)
                if projectionResult == 1:
                    contribution = np.matmul(plusVector, vs[np.mod(copy + 1, n)][alpha] / 2)
                else:
                    minusVector = (vs[copy][alpha] - y * vs[np.mod(copy + 1, n)][alpha]) / 2
                    contribution = np.matmul(minusVector, vs[np.mod(copy + 1, n)][alpha] / 2)
                result *= contribution * 2  # We chose between two options
            psiCopy = bops.shiftWorkingSite(psiCopy, alpha, '<<')
        bops.removeState(psiCopy)
    return result


# Assuming the working site of psi is already site
def applyProjector(psi: List[tn.Node], toProject: np.array, site:int):
    localDM = bops.multiContraction(psi[site], psi[site], '02', '02*').tensor
    projectionProbability = np.trace(np.matmul(localDM, toProject))
    if np.random.uniform(0, 1) < projectionProbability:
        psi[site] = bops.multNode(bops.permute(bops.multiContraction(
            psi[site], tn.Node(toProject), '1', '0', cleanOr1=True, cleanOr2=True), [0, 2, 1]),
            1 / np.sqrt(projectionProbability))
        res = 1
    else:
        psi[site] = bops.multNode(bops.permute(bops.multiContraction(
            psi[site], tn.Node(np.eye(d)  - toProject),
            '1', '0', cleanOr1=True, cleanOr2=True), [0, 2, 1]),
            1 / np.sqrt(1 - projectionProbability))
        res = 0
    return res

def getMeasuredVector(psi, toMeasure, site):
    localDM = bops.multiContraction(psi[site], psi[site], '02', '02*').tensor
