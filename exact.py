import numpy as np
from scipy.linalg import expm
from matplotlib import pyplot as plt


N = 4
T = 1
Omega = 1/T
delta = Omega
J = delta
Nu = 1000
alpha = 1.5
sigmaX = np.zeros((2, 2))
sigmaX[1,  0] = 1
sigmaX[0,  1] = 1
sigmaZ = np.zeros((2, 2))
sigmaZ[0, 0] = 1
sigmaZ[1, 1] = -1
sigmaY = -1j * np.matmul(sigmaZ, sigmaX)


def getBasicHamiltonian():
    H = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(N):
        if i > 0:
            curr = np.kron(np.eye(2**i), sigmaX)
        else:
            curr = sigmaX
        if i < N-1:
            H += Omega * np.kron(curr, np.eye(2**(N - 1 - i)))
        else:
            H += Omega * curr
    for i in range(1, N):
        for j in range(i-1, i):
            if j > 0:
                curr = np.kron(np.eye(2**j), sigmaX)
            else:
                curr = sigmaX
            if i > j + 1:
                curr = np.kron(curr, np.eye(2**(i - j - 1)))
            curr = np.kron(curr, sigmaX)
            if i < N - 1:
                curr = np.kron(curr, np.eye(2**(N - 1 - i)))
            H += (J / (i - j)**alpha) * curr
    return H


def getRandomPart():
    H = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(N):
        if i > 0:
            curr = np.kron(np.eye(2**i), sigmaZ)
        else:
            curr = sigmaZ
        H += delta * np.random.randn() * np.kron(curr, np.eye(2**(N - 1 - i)))
    return H


etasL = [0.1, 0.2,  0.3, 0.5, 1, 1.5, 2, 2.5, 3]
etas = [int(N * val) for val in etasL]

rho0 = np.zeros((2**N, 2**N))
rho0[10, 10] = 0.5
rho0[11, 11] = 0
rho0[12, 12] = 0.5
# ind = 0
# for i in range(N):
#     if i % 2 == 1:
#         ind += 2**(N - 1 - i)
# rho0[ind, ind] = 1
# rho0[ind, ind] = 0.5
# ind = 0
# for i in range(N):
#     if i % 2 == 0:
#         ind += 2**(N - 1 - i)
# rho0[ind, ind] = 0.5


H0 = getBasicHamiltonian()
for eta in etas:
    puritySum = 0
    puritySum2 = 0
    for n in range(Nu):
        U = np.eye(2**N, dtype=complex)
        for j in range(eta):
            H = H0 + getRandomPart()
            U = np.matmul(U, expm(1j * T * H))
        rho = np.matmul(U, np.matmul(rho0, np.transpose(np.conj(U))))
        puritySum += abs(rho[0, 0]) ** 2
        puritySum2 += abs(rho[2**(N-1), 2**(N-1)]) ** 2
    avg = puritySum / Nu
    avg2 = puritySum2 / Nu
    plt.scatter(eta, avg * (2**N * (2**N + 1)), c='blue')
    plt.scatter(eta, avg2 * (2**N * (2**N + 1)), c='red')

plt.show()
