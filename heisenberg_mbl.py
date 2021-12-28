import numpy as np
import pickle
import sys

def bin_state(i):
    bin_i = bin(i)[2:]
    bin_i = '0' * (L - len(bin_i)) + bin_i
    return bin_i


def szsz(i, L):
    result = 1
    bin_i = bin_state(i)
    for c in range(len(bin_i) - 1):
        if bin_i[c] != bin_i[c+1]:
            result *= -1
    return result


def pe_2(psi):
    return -np.log(sum(np.abs(psi)**4))


s_z = np.array([[1, 0], [0, -1]])
s_plus = np.array([[0, 1], [0, 0]])
s_minus = np.array([[0, 0], [1, 0]])
delta = 1
L = int(sys.argv[1])
outdir = sys.argv[2]
H_basis = np.zeros((2 ** L, 2 ** L))
for i in range(2**L):
    H_basis[i, i] = szsz(i, L)
    for c in range(L - 1):
        test = i & (2**c + 2**(c+1))
        if test != 2**c + 2**(c+1) and test != 0:
            j = i ^ (2**c + 2**(c+1))
            H_basis[i, j] = 1 / 2
inds = [i for i in range(2**L) if bin(i).count('1') == L/2]

hs = [0.1 * hi for hi in range(10, 50)]
avgs = np.zeros(len(hs))
for hi in range(len(hs)):
    h = hs[hi]
    steps = 1000
    mysum = 0
    for step in range(steps):
        H = np.copy(H_basis)
        random_hs = [np.random.uniform(-h, h) for c in range(L)]
        many_body_random_potentials = [np.sum([int(bin_state(i)[c]) * random_hs[c] for c in range(L)]) for i in range(2**L)]
        H += np.diag(many_body_random_potentials)
        H = H[:, inds[:]][inds[:]]
        vals, vecs = np.linalg.eigh(H)
        relevant_vecs = list(range(int(len(inds) / 2) - 1, int(len(inds) / 2)))
        pes = np.sum([pe_2(vecs[:, i]) for i in relevant_vecs]) / len(relevant_vecs)
        mysum += pes
    avg = mysum / steps / np.log(len(inds))
    avgs[hi] = avg
with open(outdir + '/heisenberg_mbl_L_' + str(L), 'wb') as f:
    pickle.dump(avgs, f)



