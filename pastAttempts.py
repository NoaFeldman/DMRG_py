import numpy as np
import basicOperations as bops
import tensornetwork as tn
import PEPS as peps
import toricCode as toric
import randomUs
from amtplotlib import pyplot as plt


def localUnitariesPurityEstimator(N, d, probabilities, option='allPairs'):
    if option == 'allPairs':
        return d**N * sum([sum([(-d)**(-localDistance(s, sp)) * probabilities[s] * probabilities[sp] \
                        for sp in range(d**N)]) for s in range(d**N)])
    else:
        s = np.random.randint(low=0, high=len(probabilities))
        sp = np.random.randint(low=0, high=len(probabilities))
        return d ** N * (-d) ** (-localDistance(s, sp)) * probabilities[s] * probabilities[sp]


d = 2
M = 200000
rho = toric.expectedDensityMatrix(2)
avg = 0
avg2 = 0
avgs = []
avg2s = []
systemSize = 4
spaceSize = d**systemSize
step = 100
for m in range(M):
    U = randomUs.globalUnitary(4, 2)
    # U = haar_measure(spaceSize)
    uRhoU = np.matmul(U, np.matmul(rho, np.conj(np.transpose(U))))
    avg += uRhoU[0, 0]
    avg2 += uRhoU[0, 0]**2
    if m > 0 and m % step == step - 1:
        avgs.append(avg / m)
        avg2s.append(avg2 / m)
avgs = np.array(avgs)
avg2s = np.array(avg2s)
e5s = avgs * 16
e6s = avg2s * 16 * 17 - e5s**2
plt.plot([100 * i - 1 for i in range(int(M / 100))], np.abs(e6s - sum(np.linalg.eigvalsh(rho)**2)))
b = 1


systemLengths = list(range(2, 5))
estimations = []
for l in systemLengths:
    systemSize = 2 * l
    rho = toric.expectedDensityMatrix(l)
    avg = 0
    res = []
    stepsNum = systemSize * 100000
    for m in range(stepsNum):
        U = np.eye(1, dtype=complex)
        for i in range(systemSize):
            U = np.kron(U, randomUs.haar_measure(d))
        uRhoU = np.matmul(U, np.matmul(rho, np.conj(np.transpose(U))))
        probabilities = np.diag(uRhoU)
        avg += localUnitariesPurityEstimator(systemSize, d, probabilities, option='singlePair')
        if m > stepsNum / 2 and m % 100 == 99:
           res.append(avg / m)
    estimations.append(res)
b = 1


def randomMeas(rho):
    r = np.random.uniform()
    accum = 0
    for i in range(len(rho)):
        if r > accum and r < accum + rho[i, i]:
            return i
        accum += rho[i, i]
    b = 1

# https://arxiv.org/pdf/2007.06305.pdf
M = 16 * 10000
ks = [[0] * 4 for m in range(M)]
us = [0] * M

for m in range(M):
    u0 = tn.Node(randomUs.haar_measure(d))
    openB0 = bops.multiContraction(u0, bops.multiContraction(openB, u0, '5', '1*'), '1', '0')
    u1 = tn.Node(randomUs.haar_measure(d))
    openA1 = bops.multiContraction(u1, bops.multiContraction(openA, u1, '5', '1*'), '1', '0')
    u2 = tn.Node(randomUs.haar_measure(d))
    openA2 = bops.multiContraction(u2, bops.multiContraction(openA, u2, '5', '1*'), '1', '0')
    u3 = tn.Node(randomUs.haar_measure(d))
    openB3 = bops.multiContraction(u3, bops.multiContraction(openB, u3, '5', '1*'), '1', '0')
    env123 = bops.multiContraction(circle, BEnv, '12', '03')
    env23 = bops.multiContraction(env123, AEnv, '126', '013')
    env3 = bops.permute(bops.multiContraction(env23, AEnv, '430', '023'), [2, 0, 1, 3])

    us[m] = [u0.tensor, u1.tensor, u2.tensor, u3.tensor]

    rho3 = bops.multiContraction(env3, openB3, '0123', '1234')
    ks[m][3] = randomMeas(rho3.tensor / np.trace(rho3.tensor))
    if ks[m][3] == 0:
        B3 = tn.Node(openB3.tensor[0, :, :, :, :, 0])
    else:
        B3 = tn.Node(openB3.tensor[1, :, :, :, :, 1])
    env2 = bops.permute(bops.multiContraction(env23, B3, '512', '012'), [2, 3, 1, 0])
    rho2 = bops.multiContraction(env2, openA2, '0123', '1234')
    ks[m][2] = randomMeas(rho2.tensor / np.trace(rho2.tensor))
    if ks[m][2] == 0:
        A2 = tn.Node(openA2.tensor[0, :, :, :, :, 0])
    else:
        A2 = tn.Node(openA2.tensor[1, :, :, :, :, 1])
    env1 = bops.permute(bops.multiContraction(bops.multiContraction(env123, A2, '750', '023'), B3, '235', '123')
                        , [0, 1, 3, 2])
    rho1 = bops.multiContraction(env1, openA1, '0123', '1234')
    ks[m][1] = randomMeas(rho1.tensor / np.trace(rho1.tensor))
    if ks[m][1] == 0:
        A1 = tn.Node(openA1.tensor[0, :, :, :, :, 0])
    else:
        A1 = tn.Node(openA1.tensor[1, :, :, :, :, 1])
    env0 = bops.permute(
        bops.multiContraction(bops.multiContraction(bops.multiContraction(circle, A1, '34', '01'), B3, '634', '012'),
                              A2, '530', '123'), [1, 2, 3, 0])
    rho0 = bops.multiContraction(env0, openB0, '0123', '1234')
    ks[m][0] = randomMeas(rho0.tensor / np.trace(rho0.tensor))
import pickle
with open('ks_M160000', 'wb') as fp:
    pickle.dump(ks, fp)
with open('us_M160000', 'wb') as fp:
    pickle.dump(us, fp)

def singleSiteEstimate(k, u):
    proj = np.zeros((d, d), dtype=complex)
    proj[k, k] = 1
    return 3 * np.matmul(np.conj(np.transpose(u)), np.matmul(proj, u)) - np.eye(d, dtype=complex)


purity = 0
factor = M * (M-1)
for m1 in range(M):
    for m2 in range(M):
        if m2 != m1:
            curr = 1
            for i in range(4):
                curr *= np.trace(np.matmul(singleSiteEstimate(ks[m1][i], us[m1][i]),
                                           singleSiteEstimate(ks[m2][i], us[m2][i])))
            purity += curr / factor

negativity = 0
factor = M * M-1 * M-2
for m1 in range(M):
    for m2 in range(M):
        if m2 != m1:
            for m3 in range(M):
                if m3 != m1 and m3 != m2:
                    curr = 1
                    for i in range(2):
                        curr *= np.trace(np.matmul(np.matmul(singleSiteEstimate(ks[m1][i], us[m1][i]),
                                                   singleSiteEstimate(ks[m2][i], us[m2][i])),
                                                   singleSiteEstimate(ks[m3][i], us[m3][i])))
                    for i in range(2, 4):
                        curr *= np.trace(np.matmul(np.matmul(singleSiteEstimate(ks[m1][i], us[m1][i]),
                                                             singleSiteEstimate(ks[m3][i], us[m3][i])),
                                                   singleSiteEstimate(ks[m2][i], us[m2][i])))
                    negativity += curr / factor

b = 1

fourierTensor = np.zeros((4, 4), dtype=complex)
fourierTensor[0, 0] = 1
fourierTensor[0, 3] = 1
fourierTensor[1, 1] = 1 / np.sqrt(2)
fourierTensor[1, 2] = 1 / np.sqrt(2)
fourierTensor[2, 1] = -1 / np.sqrt(2)
fourierTensor[2, 2] = 1 / np.sqrt(2)
fourier = tn.Node(fourierTensor)
doubleA = tn.Node(np.kron(A.tensor, A.tensor))
fA = bops.multiContraction(bops.multiContraction(fourier, doubleA, '1', '0'), fourier, '5', '1*', cleanOr1=True)
fAEnv = tn.Node(np.trace(fA.get_tensor(), axis1=0, axis2=5))
cUp, dUp, cDown, dDown = peps.getBMPSRowOps(
    tn.Node(np.ones((2, 4, 2))), tn.Node(np.ones((2))), tn.Node(np.ones((2, 4, 2))), tn.Node(np.ones((2))), fAEnv, fAEnv, 100)
dm = peps.bmpsDensityMatrix(cUp, dUp, cDown, dDown, fAEnv, fAEnv, fA, fA, 100)
