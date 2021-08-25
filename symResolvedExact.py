import numpy as np
import matplotlib.pyplot as plt

def getMomentum(k, N):
    return np.pi * k / (N+1)

def realSpaceToDualSpace(N):
    S = np.zeros((N, N))
    for n in range(N):
        for k in range(N):
            S[n, k] = np.sin(n * getMomentum(k, N)) * np.sqrt(2 / (N + 1))
    return S

def getCiCj0Matrix(N):
    ckcq = np.zeros((N, N))
    for i in range(int(N/2)):
        ckcq[i, i] = 1
    S = realSpaceToDualSpace(N)
    cicj = np.matmul(np.matmul(S, ckcq), np.conj(np.transpose(S)))
    return cicj

NA = 20
n = 3
cicj = getCiCj0Matrix(NA * 2)
cicj = cicj[:NA, :NA]
vals = np.linalg.eigvalsh(cicj)
alphas = [np.pi / NA * i for i in range(NA)]
sFlux = np.zeros(len(alphas))
for i in range(len(alphas)):
    alpha = alphas[i]
    sFlux[i] = 1
    for v in vals:
        sFlux[i] *= (np.exp(1j * alpha) * v ** n + (1 - v) ** n)
plt.plot(alphas, sFlux)
plt.show()

Qs = np.array(range(-NA, NA, 2))
sCharge = np.zeros(len(Qs))
for i in range(len(Qs)):
    Q = Qs[i]
    for j in range(len(alphas)):
        alpha = alphas[j]
        sCharge[i] += sFlux[j] * np.exp(1j * alpha * Q)
plt.plot(Qs, sCharge)
plt.show()

