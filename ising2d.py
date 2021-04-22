import basicOperations as bops
import PEPS as peps
import tensornetwork as tn
import pickle
import numpy as np
from matplotlib import pyplot as plt


d = 2
def getBMPS(hs):
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

        def toEnvOperator(op):
            result = bops.unifyLegs(bops.unifyLegs(bops.unifyLegs(bops.unifyLegs(
                bops.permute(op, [0, 4, 1, 5, 2, 6, 3, 7]), 6, 7), 4, 5), 2, 3), 0, 1)
            tn.remove_node(op)
            return result

        AEnv = toEnvOperator(bops.multiContraction(A, A, '4', '4*'))
        BEnv = toEnvOperator(bops.multiContraction(B, B, '4', '4*'))

        steps = 30

        envOpAB = bops.permute(bops.multiContraction(AEnv, BEnv, '1', '3'), [0, 3, 2, 4, 1, 5])
        envOpBA = bops.permute(bops.multiContraction(BEnv, AEnv, '1', '3'), [0, 3, 2, 4, 1, 5])

        curr = bops.permute(bops.multiContraction(envOpBA, envOpAB, '45', '01'), [0, 2, 4, 6, 1, 3, 5, 7])

        for i in range(2):
            [C, D, te] = bops.svdTruncation(curr, [0, 1, 2, 3], [4, 5, 6, 7], '>>', normalize=True)
            curr = bops.permute(bops.multiContraction(D, C, '23', '12'), [1, 3, 0, 5, 2, 4])
            curr = bops.permute(bops.multiContraction(curr, envOpAB, '45', '01'), [0, 2, 4, 6, 1, 3, 5, 7])

        b = 1
        currAB = curr

        rowTensor = np.zeros((11, 4, 4, 11), dtype=complex)
        rowTensor[0, 3, 3, 0] = 1
        rowTensor[1, 3, 3, 2] = 1
        rowTensor[2, 3, 3, 3] = 1
        rowTensor[3, 3, 0, 4] = 1
        rowTensor[4, 0, 3, 1] = 1
        rowTensor[5, 3, 3, 6] = 1
        rowTensor[6, 3, 0, 7] = 1
        rowTensor[7, 0, 3, 8] = 1
        rowTensor[8, 3, 3, 5] = 1
        row = tn.Node(rowTensor)

        print('60')
        upRow = bops.unifyLegs(bops.unifyLegs(bops.unifyLegs(bops.unifyLegs(
            bops.permute(bops.multiContraction(row, tn.Node(currAB.tensor), '12', '04'), [0, 2, 3, 4, 7, 1, 5, 6]), 5, 6), 5, 6), 0, 1), 0, 1)
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
        print('75')
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


def getM(orderedDM: np.array, NA=4):
    M = 0
    for i in range(2**NA):
        M += orderedDM[i, i] * bin(i).count("1") - orderedDM[i, i] * (NA - bin(i).count("1"))
    return M

hs = [k * 0.1 for k in range(51)]
getBMPS(hs)