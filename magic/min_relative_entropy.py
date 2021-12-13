import numpy as np
from typing import List
from scipy.optimize import linprog
import pickle
from scipy.optimize import minimize, NonlinearConstraint
import union_jack

digs = '012345'
def int2base(x, base=3, length=None):
    if x == 0:
        res = '0'
    digits = []
    while x:
        digits.append(digs[int(x % base)])
        x = int(x / base)
    digits.reverse()
    res = ''.join(digits)
    if length is None:
        return res
    return [0] * (length - len(res)) + [int(c) for c in res]



# Each stabilizer pure state equals U|\psi>, where \psi is a computational basis state and U clifford. Therefore,
# \rho = U|psi><psi|U^\dagger = U(\sum_i P_i)U^\dagger, where P_i are Pauli strings only containing Zs and Is. So all
# stabilizer states are characterized by d^n Pauli strings that can be diagonalized together, UP_iU^\dagger for all [i]s.
# Now, we choose all (d**2 - 1)^n Pauli strings that do not contain I, and these uniquely define the set of 2^n
# commuting Pauli strings. What's left is choosing the sign +-1 to each string and we are done.
# TODO generalize for d > 2 - probably the sign needs to become omega and the commutation needs to be sorted
def get_stab_pure(d, n):
    local_us = [np.eye(d), np.array([[1, 1], [1, -1]]) / np.sqrt(2), np.array([[1, 1j], [1j, 1]]) / np.sqrt(2)]
    basis_matrices = [np.diag([0 + 0j] * i + [1 + 0j] + [0 + 0j] * (d**n - i - 1)) for i in range(d**n)]
    result = []
    for base_string in range((d**2 - 1)**n):
        string_indices = int2base(base_string, base=(d**2 - 1), length=n)
        u = np.eye(1, dtype=complex)
        for i in range(n):
            u = np.kron(u, local_us[string_indices[i]])
        for basis_i in range(len(basis_matrices)):
            result.append(np.matmul(u, np.matmul(basis_matrices[basis_i], np.conj(np.transpose(u)))))
    return result


def union_jack_symmetrize(stab_pure: List[np.array]):
    d = 2
    n = 5
    transformations = [[1, 3, 2, 4, 0],
                       [4, 0, 2, 1, 3],
                       [3, 4, 2, 0, 1],
                       [0, 1, 2, 3, 4]]
    i = 0
    result = []
    while len(stab_pure) > 0:
        print(len(stab_pure))
        curr = stab_pure[0]
        curr1 = np.zeros((d**n, d**n))
        curr2 = np.zeros((d**n, d**n))
        curr3 = np.zeros((d**n, d**n))
        for i in range(d**n):
            broken_i = int2base(i, base=d, length=n)
            for j in range(d**n):
                broken_j = int2base(j, base=d, length=n)
                curr1[sum([broken_i[k] * 2**transformations[1][k] for k in range(n)]),
                      sum([broken_j[k] * 2**transformations[1][k] for k in range(n)])] = curr[i, j]
                curr2[sum([broken_i[k] * 2**transformations[2][k] for k in range(n)]),
                      sum([broken_j[k] * 2**transformations[2][k] for k in range(n)])] = curr[i, j]
                curr3[sum([broken_i[k] * 2**transformations[3][k] for k in range(n)]),
                      sum([broken_j[k] * 2**transformations[3][k] for k in range(n)])] = curr[i, j]
        del stab_pure[0]
        if True in [np.array_equal(curr1, x) for x in stab_pure]:
            del stab_pure[[np.array_equal(curr1, x) for x in stab_pure].index(True)]
        if True in [np.array_equal(curr2, x) for x in stab_pure]:
            del stab_pure[[np.array_equal(curr2, x) for x in stab_pure].index(True)]
        if True in [np.array_equal(curr3, x) for x in stab_pure]:
            del stab_pure[[np.array_equal(curr3, x) for x in stab_pure].index(True)]
        result.append((curr + curr1 + curr2 + curr3) / 4)
    return result



# TODO impose symmetries
# TODO get coefficients for optimization

def optimization_coefficients(rho: np.array, stab_pure: List[np.array]):
    result = np.zeros(len(stab_pure))
    vals, vecs = np.linalg.eigh(rho)
    vals = [int(val > 1e-8) for val in vals]
    rho_support = np.matmul(vecs, np.matmul(np.diag(vals), np.conj(np.transpose(vecs))))
    for i in range(len(stab_pure)):
    #     result[i] = np.trace(np.matmul(rho, stab_pure[i]))
        result[i] = np.trace(np.matmul(rho_support, stab_pure[i]))
    return result

def get_min_relative_entropy(rho, union_jack=True, d=2, n=5):
    # pure = get_stab_pure(d, n)
    # if union_jack:
    #     pure = union_jack_symmetrize(pure)
    with open('magic/results/stab_pure_union_jack', 'rb') as f:
        pure = pickle.load(f)
    # TODO get projection to support
    obj_basis = optimization_coefficients(rho, pure)


# TODO min relative entropy with Tr(rhologrho - Trrhologsigma using
#  https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html
# sigma = \sum_i x_i S_i ...
