import numpy as np
import pickle
import tdvp
import sys
import os
import tensornetwork as tn
import basicOperations as bops
import scipy.linalg as linalg
import time

def numberToBase(n, b, N):
    if n == 0:
        return [0] * N
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return digits[::-1] + [0] * (N-len(digits))


def pauli_strings_to_stabilizers(n):
    H = np.array([[1, 1], [1, -1]])
    hadmard_n = np.eye(1)
    for i in range(int(np.log(n)/np.log(2))):
        hadmard_n = np.kron(hadmard_n, H)
    pauli_strings = [[int(c) + 1 for c in numberToBase(i, 3, n)] for i in range(3**n)]
    row = []
    col = []
    data = []
    
pauli_strings_to_stabilizers(4)