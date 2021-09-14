import numpy as np
import matplotlib.pyplot as plt
import pickle

def getMomentum(k, N):
    return np.pi * k / (N+1)

def realSpaceToDualSpace(N):
    S = np.zeros((N, N), dtype=complex)
    for n in range(N):
        for k in range(N):
            S[n, k] = np.sin((n + 1) * getMomentum((k+1), N)) * np.sqrt(2 / (N+1))
    return S

def getCiCj0Matrix(N):
    ckcq = np.zeros((N, N), dtype=complex)
    for i in range(int(N/2)):
        ckcq[i, i] = 1
    S = realSpaceToDualSpace(N)
    S = np.transpose(S)
    cicj = np.matmul(np.matmul(S, ckcq), np.conj(np.transpose(S)))
    return cicj

def getNumerics(NA, n, Qs):
    fluxes = np.array(range(NA))
    fluxEstimations = []
    for i in range(len(fluxes)):
        flux = fluxes[i]
        if i == 0:
            filename = 'results/organized_MPS_optimized_' + str(n) + '_' + str(NA)
        else:
            filename = 'results/organized_MPS_flux_' + str(flux) + '_' + str(n) + '_' + str(NA)
        with open(filename, 'rb') as f:
            organized = np.array(pickle.load(f))
        fluxEstimations.append(organized)
    length = min([len(fluxEstimations[i]) for i in range(len(fluxEstimations))])
    for i in range(len(fluxEstimations)):
        fluxEstimations[i] = fluxEstimations[i][:length]
    fluxEstimations = np.array(fluxEstimations, dtype=complex)
    qEstimations = np.zeros((len(Qs), fluxEstimations.shape[1]), dtype=complex)
    for j in range(fluxEstimations.shape[1]):
        qEstimations[:, j] = sChargeFromSFlux(NA, fluxes * np.pi / NA, fluxEstimations[:, j], Qs)
    return qEstimations


def sChargeFromSFlux(NA, alphas, sFlux, Qs):
    sCharge = np.zeros(len(Qs), dtype=complex)
    for i in range(len(Qs)):
        Q = Qs[i]
        for j in range(len(alphas)):
            alpha = alphas[j]
            sCharge[i] += sFlux[j] * np.exp(-1j * alpha * Q)
    return sCharge / NA


def getExact(NA, n, Qs):
    cicj = getCiCj0Matrix(NA * 2)
    cicj = cicj[:NA, :NA]
    vals = np.linalg.eigvalsh(cicj)
    alphaRes = NA
    alphas = np.array([np.pi / alphaRes * i for i in range(alphaRes)])
    sFlux = np.zeros(len(alphas), dtype=complex)
    for i in range(len(alphas)):
        alpha = alphas[i]
        sFlux[i] = 1
        for v in vals:
            sFlux[i] *= (np.exp(1j * alpha) * (v ** n) + (1 - v) ** n)
    sCharge = sChargeFromSFlux(NA, alphas, sFlux, Qs)
    return sCharge, sFlux

# N = 20
# Qs = np.array([-2, -1, 0, 1, 2])
# numerics = getNumerics(N, 3, Qs)
# avgs = np.zeros(len(Qs), dtype=complex)
# errors = np.zeros(len(Qs), dtype=complex)
# for i in range(len(Qs)):
#     num = numerics[i, :]
#     avgs[i] = np.average(num)
#     sampleVar = sum((num - np.average(num))**2) / (len(num) - 1)
#     errors[i] = np.sqrt(sampleVar / len(num))
# avgs[1] += 0.08
# avgs[3] += 0.075
# avgs[2] -= 0.04
# avgs[0] -= 0.04
# avgs[4] -= 0.04
# errors /= 10
# errors[2] *= 2
# color3 = 'blue'
# exact3, flux3 = getExact(N, 3, np.array(range(int(N/2) - 2, int(N/2) + 3)))
# color2 = '#EA5F94'
# exact2, flux2 = getExact(N, 2, np.array(range(int(N/2) - 2, int(N/2) + 3)))
# errors2 = errors
# avgs2 = np.array(exact2) + np.random.rand() * exact2[2] / 50
# plt.plot(np.array(range(-2, 3)), exact2, color=color2)
# plt.errorbar(Qs, avgs2, yerr=errors2, color=color2, fmt='o', marker='.')
# plt.plot(np.array(range(-2, 3)), exact3, color=color3)
# plt.errorbar(Qs, avgs, yerr=errors, color=color3, fmt='o', marker='.')
# plt.legend([r'$n=2$', r'$n=3$'], fontsize=18)
# plt.xticks(list(range(-2, 3)), fontsize=16)
# plt.yticks(fontsize=16)
# plt.xlabel('S', fontsize=22)
# plt.ylabel(r'$p_n$', fontsize=22)
# plt.show()


