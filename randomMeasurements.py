import tensornetwork as tn
import numpy as np
import basicOperations as bops
import scipy

"""A Random matrix distributed with Haar measure"""
""" from https://arxiv.org/pdf/math-ph/0609050.pdf """
def haar_measure(d: int, axisName: str):
    z = (np.random.randn(d, d) + 1j * np.random.randn(d, d)) / np.sqrt(2.0)
    q, r = scipy.linalg.qr(z)
    diag = np.diagonal(r)
    ph = diag / np.absolute(diag)
    q = np.multiply(q, ph, q)
    return tn.Node(q, axis_names=[axisName, axisName + '*'], backend=None)


def randomMeasurement(psi, startInd, endInd):
    d = psi[0].tensor.shape[1]
    res = [-1] * (endInd - startInd)
    psiCopy = bops.copyState(psi)
    for k in [len(psiCopy) - 1 - i for i in range(len(psiCopy) - startInd - 1)]:
        psiCopy = bops.shiftWorkingSite(psiCopy, k, '<<')
    for i in range(startInd, endInd):
        rho = bops.multiContraction(psiCopy[i], psiCopy[i], '02', '02*')
        measurement = np.random.uniform(low=0, high=1)
        covered = 0
        for s in range(len(rho.tensor)):
            if covered < measurement < covered + rho.tensor[s, s]:
                res[i - startInd] = s
                break
            covered += rho.tensor[s, s]
        projectorTensor = np.zeros((d, d), dtype=complex)
        projectorTensor[res[i - startInd], res[i - startInd]] = 1
        projector = tn.Node(projectorTensor, backend=None)
        bops.applySingleSiteOp(psiCopy, projector, i)
        psiCopy = bops.shiftWorkingSite(psiCopy, i, '>>')
        psiCopy[i + 1].tensor /= np.sqrt(bops.getOverlap(psiCopy, psiCopy))
        tn.remove_node(rho)
    bops.removeState(psiCopy)
    return res


