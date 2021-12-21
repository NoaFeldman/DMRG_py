import numpy as np
from typing import List
from scipy.optimize import linprog
import pickle
from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint, Bounds
import magic.basicDefs as basic

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
    independent = []
    for base_string in range((d**2 - 1)**n):
        string_indices = int2base(base_string, base=(d**2 - 1), length=n)
        u = np.eye(1, dtype=complex)
        for i in range(n):
            u = np.kron(u, local_us[string_indices[i]])
        for basis_i in range(len(basis_matrices)):
            curr = np.matmul(u, np.matmul(basis_matrices[basis_i], np.conj(np.transpose(u))))
            result.append(curr)
            if 2 not in string_indices:
                independent.append(len(result) - 1)
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
    pure = get_stab_pure(d, n)
    # if union_jack:
    #     pure = union_jack_symmetrize(pure)
    # with open('magic/results/stab_pure_union_jack', 'rb') as f:
    #     pure = pickle.load(f)
    obj_func = lambda x: relative_entropy_obj_func(rho, pure, x)
    x0 = np.zeros(len(pure))
    x0[0] = 1
    # Add constraint \sum_i x_i = 1
    constraint_mat = np.zeros((len(x0), len(x0)))
    constraint_mat[0, :] = np.ones(len(x0))
    constraint_max = np.zeros(len(x0))
    constraint_max[0] = 1
    constraint_min = np.zeros(len(x0))
    constraint_min[0] = 1
    linear_constraint = LinearConstraint(constraint_mat, constraint_min, constraint_max)
    bounds = Bounds(np.zeros(len(x0)), np.ones(len(x0)))
    res = minimize(obj_func, x0, method='trust-constr', bounds=bounds,
                   options={'verbose': 1}, constraints=[linear_constraint])
    b = 1

#  https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html

def relative_entropy_obj_func(rho, pure, x, d=2, n=5):
    sigma = sum([x[i] * pure[i] for i in range(len(pure))])
    return np.trace(np.matmul(rho, sigma)) / np.trace(np.linalg.matrix_power(sigma, 2))


def get_full_basis(d, n):
    if d == 2:
        basis = []
        single_site_ops = [np.eye(2), basic.pauli2X, basic.pauli2Y, basic.pauli2Z]
        for string in range(4**n):
            string_indices = int2base(string, base=4, length=n)
            curr = np.eye(1)
            for site in range(n):
                curr = np.kron(single_site_ops[string_indices[site]], curr)
            basis.append(curr)
        return basis


def robustness_constraints(rho, pure, d=2, n=5):
    full_basis = get_full_basis(d, n)
    # This is effectively len(full_basis) x len(pure), but the constraints are required to be squared.
    x_rho = np.zeros(len(pure))
    for i in range(len(full_basis)):
        x_rho[i] = np.trace(np.matmul(full_basis[i], rho))
    constraint_mat = np.zeros((len(pure), len(pure)))
    for pure_ind in range(len(pure)):
        for i in range(len(full_basis)):
            constraint_mat[i, pure_ind] = np.trace(np.matmul(full_basis[i], pure[pure_ind]))
    return constraint_mat, x_rho


def robustness_obj_func(x):
    return sum(np.abs(x))


def get_robustness(rho, d=2, n=5):
    pure = get_stab_pure(d, n)
    constraint_mat, x_rho = robustness_constraints(rho, pure)
    linear_constraint = LinearConstraint(constraint_mat, x_rho, x_rho)
    res = minimize(robustness_obj_func, x_rho, method='trust-constr',
                   constraints=[linear_constraint],
                   options={'verbose': 1})
    b = 1
