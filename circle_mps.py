import os.path
import basicOperations as bops
import numpy as np
import DMRG as dmrg
import tensornetwork as tn
import pickle
import sys
from os import path


def get_H_terms(N, onsite_term, neighbor_term, d=2):
    onsite_terms = [np.kron(onsite_term, onsite_term) for i in range(int(N / 2))]
    onsite_terms[0] += neighbor_term
    neighbor_terms = [np.kron(neighbor_term, neighbor_term) for i in range(int(N/2) - 1)]
    if N % 2 == 0:
        onsite_terms[-1] += neighbor_term
    else:
        neighbor_terms += [np.kron(neighbor_term, np.eye(d**2))
                + np.kron(neighbor_term, np.eye(d**2)).reshape([d] * 4).transpose([1, 0, 2, 3]).reshape([d**2] * 2)]
    return onsite_terms, neighbor_terms