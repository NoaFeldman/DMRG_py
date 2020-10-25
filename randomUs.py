import numpy as np
import basicOperations as bops
import tensornetwork as tn
import scipy


"""A Random matrix distributed with Haar measure"""
def haar_measure(n):
    z = (np.random.randn(n, n) + 1j * np.random.randn(n, n)) / np.sqrt(2.0)
    q,r = scipy.linalg.qr(z)
    d = np.diagonal(r)
    ph = d / np.absolute(d)
    q = np.multiply(q,ph,q)
    return q


# create a global unitary from 2 layers of nearest neighbor unitaries
def globalUnitary(N, d, numberOfLayers=2):
    U = np.eye(d**N)
    for i in range(numberOfLayers):
        u01 = np.kron(haar_measure(d**2), np.eye(d**2, dtype=complex))
        u02 = np.reshape(
            np.transpose(np.reshape(np.kron(haar_measure(d ** 2), np.eye(d ** 2, dtype=complex)), [d] * 2 * N),
                         [0, 2, 1, 3, 4, 6, 5, 7]), [d ** N, d ** N])
        u23 = np.kron(np.eye(d**2, dtype=complex), haar_measure(d**2))
        u13 = np.reshape(np.transpose(np.reshape(np.kron(haar_measure(d**2), np.eye(d**2, dtype=complex)), [d] * 2 * N),
                                      [2, 0, 3, 1, 6, 4, 7, 5]), [d**N, d**N])
        U = np.matmul(U, np.matmul(u01, np.matmul(u02, np.matmul(u23, u13))))
    return U


