import numpy as np
import matplotlib.pyplot as plt
import pickle

def getMomentum(k, N):
    return np.pi * k / (N+1)

def realSpaceToDualSpace(N):
    S = np.zeros((N, N), dtype=complex)
    for n in range(N):
        for k in range(N):
            S[n, k] = np.sin((n + 1) * getMomentum((k+1), N)) * np.sqrt(2 / N)
    return S

def getCiCj0Matrix(N):
    ckcq = np.zeros((N, N), dtype=complex)
    for i in range(int(N/2)):
        ckcq[i, i] = 1
    S = realSpaceToDualSpace(N)
    cicj = np.matmul(np.matmul(S, ckcq), np.conj(np.transpose(S)))
    return cicj

def analyzeNumerics(NA, n):
    fluxes = np.array(range(NA))
    estimations = np.zeros(len(fluxes), dtype=complex)
    for i in range(len(fluxes)):
        flux = (fluxes[i] * 3) % 20
        if i == 0:
            filename = 'results/organized_MPS_optimized_' + str(n) + '_' + str(NA)
        else:
            filename = 'results/organized_MPS_flux_' + str(flux) + '_' + str(n) + '_' + str(NA)
        with open(filename, 'rb') as f:
            organized = np.array(pickle.load(f))
        estimations[i] = np.average(organized)
    return estimations


NA = 8
n = 1
cicj = getCiCj0Matrix(NA * 2)
cicj = cicj[:NA, :NA]
vals = np.linalg.eigvalsh(cicj)
alphaRes = NA
alphas = [np.pi / alphaRes * i for i in range(1, alphaRes)]
sFlux = np.zeros(len(alphas), dtype=complex)
for i in range(len(alphas)):
    alpha = alphas[i]
    sFlux[i] = 1
    for v in vals:
        sFlux[i] *= (np.exp(1j * alpha) * (v ** n) + (1 - v) ** n)
# plt.plot(alphas, np.real(sFlux))
# plt.plot(alphas, np.imag(sFlux))
# numerics = analyzeNumerics(NA, n)
# plt.plot(alphas, numerics)
# plt.show()

def sChargeFromSFlux(alphas, sFlux, NA, step):
    Qs = np.array(range(-int(step * NA/2), int(step * NA/2)+step, step))
    # Qs = np.array(range(-100, 100)) / 100 * NA
    sCharge = np.zeros(len(Qs), dtype=complex)
    for i in range(len(Qs)):
        Q = Qs[i]
        for j in range(len(alphas)):
            alpha = alphas[j]
            sCharge[i] += sFlux[j] * np.exp(-1j * alpha * Q)
    return Qs, sCharge / NA

Qs, sCharge = sChargeFromSFlux(alphas, sFlux, NA, 2)
plt.plot(Qs, np.real(sCharge))
# Qs, sCharge = sChargeFromSFlux(alphas, numerics, NA, 1)
# plt.plot(Qs, np.real(sCharge))
print(sum(sCharge))
print(sFlux[0])
plt.show()
#

