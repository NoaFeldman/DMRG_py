import numpy as np
import tensornetwork as tn

d = 3
omega = np.exp(1j * 2 * np.pi / d)
pauliX = np.zeros((d, d), dtype=complex)
pauliX[0, -1] = 1
pauliX[1, 0] = 1
pauliX[2, 1] = 1
Px = tn.Node(pauliX)
pauliZ = np.diag(np.array([omega ** j for j in range(d)]))
Pz = tn.Node(pauliZ)

pauli2X = np.zeros((2, 2), dtype=complex)
pauli2X[0, 1] = 1
pauli2X[1, 0] = 1
pauli2Z = np.diag([1, -1])
pauli2Y = np.matmul(pauli2Z, pauli2X) / 1j

def getPauliMatrices(d):
    if d == 2:
        return([np.eye(2, dtype=complex), pauli2X, pauli2Y, pauli2Z])
    elif d == 3:
        # TODO
        return 0