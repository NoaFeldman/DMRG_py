import tensornetwork as tn
import numpy as np
import basicOperations as bops
import DMRG as dmrg
from typing import List

d = 2
randomPossibilities = 2

def singleHaarEstimation(psi: List[tn.Node], us: List[tn.Node], n: int, NA: int):
    psiCopy = bops.copyState(psi)
    projector0 = np.zeros((2, 2))
    projector0[0, 0] = 1
    projector1 = np.zeros((2, 2))
    projector1[1, 1] = 1
    measurementResults = np.array(NA)
    estimation = 1
    for i in range(NA):
        psiCopy[i] = bops.permute(bops.multiContraction(psiCopy[i], us[i], '1', '1', cleanOr1=True), [0, 2, 1])
        measurementResults[i] = makeMeasurement(psiCopy, i, projector0)
        if measurementResults:
            resultProjector = projector0
        else:
            resultProjector = projector1
        localEstimation = np.matmul(np.matmul(np.conj(np.transpose(us[i].tensor)), resultProjector), us[i].tensor) \
            - np.eye(2)
        estimation *= np.trace(localEstimation**4)
        psiCopy = bops.shiftWorkingSite(psiCopy, i, '<<')
    return estimation


def expectationValues(psi: List[tn.Node], vs: List[List[np.array]]):
    vs = np.round(vs, 10)
    n = len(vs)
    NA = len(vs[0])
    result = 1
    # result = np.eye(2, dtype=complex)
    for copy in range(n):
        psiCopy = bops.copyState(psi)
        for alpha in range(NA - 1, -1, -1):
            overlap = np.matmul(vs[copy][alpha], np.conj(vs[np.mod(copy + 1, n)][alpha]))
            toEstimate = np.outer(vs[copy][alpha], np.conj(vs[np.mod(copy + 1, n)][alpha]))
            if np.abs(np.round(overlap, 8)) == 2:
                toMeasure = toEstimate
            else:
                hermitianComponent = np.random.randint(2)
                if hermitianComponent:
                    toMeasure = (toEstimate + np.conj(np.transpose(toEstimate)))
                else:
                    toMeasure = (toEstimate - np.conj(np.transpose(toEstimate)))
            psiCopy[alpha] = bops.permute(bops.multiContraction(psiCopy[alpha], tn.Node(toMeasure), \
                                                   '1', '1'), [0, 2, 1])
            psiCopy = bops.shiftWorkingSite(psiCopy, alpha, '<<')
        result *= bops.getOverlap(psiCopy, psi)
        # result = np.matmul(result, toMeasure)
        bops.removeState(psiCopy)
    return result


# vs[i][j] = random vector for copy i, site j
def singleMeasurement(psi: List[tn.Node], vs: List[List[np.array]]):
    vs = np.round(vs, 10)
    n = len(vs)
    NA = len(vs[0])
    result = 1
    for copy in range(n):
        psiCopy = bops.copyState(psi)
        for alpha in range(NA - 1, -1, -1):
            toEstimate = np.outer(vs[copy][alpha], np.conj(vs[np.mod(copy+1, n)][alpha]))
            hermitianComponent = np.random.randint(2)
            if hermitianComponent:
                toMeasure = (toEstimate + np.conj(np.transpose(toEstimate)))
            else:
                toMeasure = (toEstimate - np.conj(np.transpose(toEstimate))) / 1j
            measureVals, measureVecs = np.linalg.eigh(toMeasure)
            projector = np.outer(measureVecs[:, 0], np.conj(measureVecs[:, 0]))
            measResult = makeMeasurement(psiCopy, alpha, projector)
            if measResult:
                if np.abs(measureVals[0]) < 1e-8:
                    return 0
                result *= measureVals[0]
            else:
                if np.abs(measureVals[1]) < 1e-8:
                    return 0
                result *= measureVals[1]
            if not hermitianComponent:
                result *= 1j
            psiCopy = bops.shiftWorkingSite(psiCopy, alpha, '<<')
        bops.removeState(psiCopy)
    return result

def getExpectationValue(psi, site, op):
    localDM = bops.multiContraction(psi[site], psi[site], '02', '02*').tensor
    result = np.trace(np.matmul(localDM, op))
    return result

# Assuming the working site of psi is already site
def makeMeasurement(psi, site, toProject):
    localDM = bops.multiContraction(psi[site], psi[site], '02', '02*').tensor
    projectionProbability = np.trace(np.matmul(localDM, toProject)) / np.trace(localDM)
    # Project to the measured vector
    if np.random.uniform(0, 1) < projectionProbability:
        psi[site] = bops.permute(bops.multiContraction(
            psi[site], tn.Node(toProject), '1', '1', cleanOr1=True, cleanOr2=True), [0, 2, 1])
        res = 1
    else:
        psi[site] = bops.permute(bops.multiContraction(
            psi[site], tn.Node(np.eye(d) - toProject), '1', '1', cleanOr1=True, cleanOr2=True), [0, 2, 1])
        res = 0
    return res


def getMeasuredVector(psi, toMeasure, site):
    localDM = bops.multiContraction(psi[site], psi[site], '02', '02*').tensor
