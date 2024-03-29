import basicOperations as bops
import PEPS as peps
import tensornetwork as tn
import pickle
import numpy as np
from matplotlib import pyplot as plt


d = 2
def getBMPS(hs, steps):
    for h in hs:
        h = np.round(h, 1)
        if int(h) == h:
            h = int(h)
        with open('ising/origTensors/ising_field_' + str(h) + '_A', 'rb') as f:
            ATensor = pickle.load(f)
        with open('ising/origTensors/ising_field_' + str(h) + '_B', 'rb') as f:
            BTensor = pickle.load(f)
        A = tn.Node(ATensor)
        A = bops.permute(A, [1, 2, 3, 0, 4])
        B = tn.Node(BTensor)
        B = bops.permute(B, [1, 2, 3, 0, 4])
        print('23')

        def toEnvOperator(op):
            result = bops.unifyLegs(bops.unifyLegs(bops.unifyLegs(bops.unifyLegs(
                bops.permute(op, [0, 4, 1, 5, 2, 6, 3, 7]), 6, 7), 4, 5), 2, 3), 0, 1)
            tn.remove_node(op)
            return result

        AEnv = toEnvOperator(bops.multiContraction(A, A, '4', '4*'))
        BEnv = toEnvOperator(bops.multiContraction(B, B, '4', '4*'))

        envOpAB = bops.permute(bops.multiContraction(AEnv, BEnv, '1', '3'), [0, 3, 2, 4, 1, 5])
        envOpBA = bops.permute(bops.multiContraction(BEnv, AEnv, '1', '3'), [0, 3, 2, 4, 1, 5])

        curr = bops.permute(bops.multiContraction(envOpBA, envOpAB, '45', '01'), [0, 2, 4, 6, 1, 3, 5, 7])

        for i in range(2):
            [C, D, te] = bops.svdTruncation(curr, [0, 1, 2, 3], [4, 5, 6, 7], '>>', normalize=True)
            curr = bops.permute(bops.multiContraction(D, C, '23', '12'), [1, 3, 0, 5, 2, 4])
            curr = bops.permute(bops.multiContraction(curr, envOpAB, '45', '01'), [0, 2, 4, 6, 1, 3, 5, 7])
        print('43')

        b = 1
        currAB = curr

        # rowTensor = np.zeros((13, 4, 4, 13), dtype=complex)
        # for i in range(13 * 4 * 4):
        #     rowTensor[np.random.randint(0, 13), np.random.randint(0, 4), np.random.randint(0, 4), np.random.randint(0, 13)] *= np.random.randint(0, 2) * 2 - 1
        # row = tn.Node(rowTensor)

        # upRow = bops.unifyLegs(bops.unifyLegs(bops.unifyLegs(bops.unifyLegs(
        #     bops.permute(bops.multiContraction(row, tn.Node(currAB.tensor), '12', '04'), [0, 2, 3, 4, 7, 1, 5, 6]), 5, 6), 5, 6), 0, 1), 0, 1)

        with open('ising/bmpsResults_' + str(2.9), 'rb') as f:
            [upRow, downRow, leftRow, rightRow, openA, openB, A, B] = pickle.load(f)
        [C, D, te] = bops.svdTruncation(upRow, [0, 1], [2, 3], '>>', normalize=True)
        upRow = bops.multiContraction(D, C, '2', '0')
        [cUp, dUp, te] = bops.svdTruncation(upRow, [0, 1], [2, 3], '>>', normalize=True)
        GammaC, LambdaC, GammaD, LambdaD = peps.getBMPSRowOps(cUp, tn.Node(np.ones(cUp[2].dimension)), dUp,
                                                    tn.Node(np.ones(dUp[2].dimension)), AEnv, BEnv, steps)
        cUp = bops.multiContraction(GammaC, LambdaC, '2', '0', isDiag2=True)
        dUp = bops.multiContraction(GammaD, LambdaD, '2', '0', isDiag2=True)
        upRow = bops.multiContraction(cUp, dUp, '2', '0')
        downRow = bops.copyState([upRow])[0]
        rightRow = peps.bmpsCols(upRow, downRow, AEnv, BEnv, steps, option='right', X=upRow)
        leftRow = peps.bmpsCols(upRow, downRow, AEnv, BEnv, steps, option='left', X=upRow)
        circle = bops.multiContraction(
            bops.multiContraction(bops.multiContraction(upRow, rightRow, '3', '0'), upRow, '5', '0'), leftRow, '70',
            '03')

        openA = tn.Node(np.transpose(np.reshape(np.kron(A.tensor, np.conj(A.tensor)), [d**2, d**2, d**2, d**2, d, d]), [4, 0, 1, 2, 3, 5]))
        openB = tn.Node(np.transpose(np.reshape(np.kron(B.tensor, np.conj(B.tensor)), [d**2, d**2, d**2, d**2, d, d]), [4, 0, 1, 2, 3, 5]))
        ABNet = bops.permute(
            bops.multiContraction(bops.multiContraction(openB, openA, '2', '4'),
                                  bops.multiContraction(openA, openB, '2', '4'), '28', '16',
                                  cleanOr1=True, cleanOr2=True),
            [1, 5, 6, 13, 14, 9, 10, 2, 0, 4, 8, 12, 3, 7, 11, 15])
        dm = bops.multiContraction(circle, ABNet, '01234567', '01234567')
        ordered = np.round(np.reshape(dm.tensor, [16, 16]), 14)
        ordered /= np.trace(ordered)
        print(h, getM(ordered))
        with open('ising/bmpsResults_' + str(h), 'wb') as f:
            pickle.dump([upRow, downRow, leftRow, rightRow, openA, openB, A, B], f)
        print(h)


def getM(orderedDM: np.array, NA=4):
    M = 0
    for i in range(2**NA):
        M += orderedDM[i, i] * bin(i).count("1") - orderedDM[i, i] * (NA - bin(i).count("1"))
    return M

# getBMPS([2.7, 2.8], 20)

hs = [0.1 * k for k in range(51)]
Ms = [0] * len(hs)
p2s = [0] * len(hs)
p3s = [0] * len(hs)
p4s = [0] * len(hs)
for i in range(len(hs)):
    h = np.round(hs[i], 1)
    if h == int(h):
        h = int(h)
    with open('ising/bmpsResults_' + str(h), 'rb') as f:
        [upRow, downRow, leftRow, rightRow, openA, openB, A, B] = pickle.load(f)
    circle = bops.multiContraction(
        bops.multiContraction(bops.multiContraction(upRow, rightRow, '3', '0'), upRow, '5', '0'), leftRow, '70',
        '03')

    openA = tn.Node(
        np.transpose(np.reshape(np.kron(A.tensor, np.conj(A.tensor)), [d ** 2, d ** 2, d ** 2, d ** 2, d, d]),
                     [4, 0, 1, 2, 3, 5]))
    openB = tn.Node(
        np.transpose(np.reshape(np.kron(B.tensor, np.conj(B.tensor)), [d ** 2, d ** 2, d ** 2, d ** 2, d, d]),
                     [4, 0, 1, 2, 3, 5]))
    ABNet = bops.permute(
        bops.multiContraction(bops.multiContraction(openB, openA, '2', '4'),
                              bops.multiContraction(openA, openB, '2', '4'), '28', '16',
                              cleanOr1=True, cleanOr2=True),
        [1, 5, 6, 13, 14, 9, 10, 2, 0, 4, 8, 12, 3, 7, 11, 15])
    dm = bops.multiContraction(circle, ABNet, '01234567', '01234567')
    ordered = np.round(np.reshape(dm.tensor, [16, 16]), 14)
    ordered /= np.trace(ordered)
    Ms[i] = getM(ordered)
    p2s[i] = np.trace(np.linalg.matrix_power(ordered, 2))
    p3s[i] = np.trace(np.linalg.matrix_power(ordered, 3))
    p4s[i] = np.trace(np.linalg.matrix_power(ordered, 4))
plt.scatter(hs, np.abs(Ms))
plt.xlabel('h')
plt.ylabel('|M|')
plt.show()
plt.plot(hs, p2s)
plt.plot(hs, p3s)
plt.plot(hs, p4s)
plt.xlabel('h')
plt.legend([r'$p_2$', r'$p_3$', r'$p_4$'])
plt.title(r'2nd Renyi moment for a $2\times2$ system (calculated with the explicit RDM)')
plt.show()

