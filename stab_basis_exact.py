import numpy as np
import magicRenyi as mr
import basicOperations as bops
import matplotlib.pyplot as plt
import os.path as path
import pickle


d = 2
I = np.eye(d)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1, -1])
paulis = [I, X, Y, Z]

def get_gs_stab_renyi_2(L, H, option='optimize', basis=[]):
    vals, vecs = np.linalg.eigh(H)
    gs = vecs[:, np.argmin(vals)]
    psi = bops.vector_to_mps(gs, d, L)
    if option == 'optimize':
        res = mr.getSecondRenyi_optimizedBasis(psi, d)[0]
    elif option == 'basis':
        res = mr.getSecondRenyi_basis(psi, d, basis[0], basis[1], basis[2])
    else:
        res = mr.getSecondRenyi(psi, d)
    return res


# Eq. (4) here https://arxiv.org/pdf/2209.10541.pdf
def xx_term(L):
    H = np.zeros((d**L, d**L), dtype=complex)
    for l in range(L-1):
        H += np.kron(np.eye(d**l), np.kron(np.kron(X, X), np.eye(d**(L - l - 2))))
    H += np.kron(X, np.kron(np.eye(d**(L - 2)), X))
    return H


def tfim_term(L):
    H = np.zeros((d**L, d**L), dtype=complex)
    for l in range(L):
        H += np.kron(np.eye(d**l), np.kron(Z, np.eye(d**(L - l - 1))))
    return H


def mim_term(L):
    H = np.zeros((d**L, d**L), dtype=complex)
    for l in range(L):
        H += np.kron(np.eye(d**l), np.kron(np.sqrt(1/2) * Z + np.sqrt(2) * X, np.eye(d**(L - l - 1))))
    return H


def cim_term(L):
    H = np.zeros((d ** L, d ** L), dtype=complex)
    for l in range(L - 2):
        H += np.kron(np.eye(d ** l), np.kron(np.kron(Y, np.kron(Z, Y)),
                                             np.eye(d ** (L - l - 3))))
    H += np.kron(Y, np.kron(np.eye(d ** (L - 3)), np.kron(Y, Z)))
    H += np.kron(np.kron(Z, Y), np.kron(np.eye(d ** (L - 3)), Y))
    return H


def get_H(L, lambd, J, model='tfim'):
    H = J * xx_term(L)
    if model == 'tfim':
        return H + lambd * tfim_term(L)
    elif model == 'cim':
        return H + lambd * cim_term(L)
    elif model == 'mim':
        return H + lambd * mim_term(L)


def gradient_descent(H, L, accuracy=1e-2, opt='min'):
    learning_rate = 1e-2
    theta, phi, eta = [0, 0, 0]
    M = get_gs_stab_renyi_2(L, H, option='basis', basis=[theta, phi, eta])
    M_d_theta = 1
    M_d_phi = 1
    M_d_eta = 1
    theta_step = learning_rate
    phi_step = learning_rate
    eta_step = learning_rate
    for si in range(int(1e3)):
        new_m_d_theta = (get_gs_stab_renyi_2(L, H, option='basis', basis=[theta + theta_step, phi, eta]) - M) / theta_step
        new_m_d_phi = (get_gs_stab_renyi_2(L, H, option='basis', basis=[theta, phi + phi_step, eta]) - M) / phi_step
        new_m_d_eta = (get_gs_stab_renyi_2(L, H, option='basis', basis=[theta, phi, eta + eta_step]) - M) / eta_step
        # if np.abs(new_m_d_theta) > np.abs(M_d_theta):
        #     theta_step /= 2
        # if np.abs(new_m_d_phi) > np.abs(M_d_phi):
        #     phi_step /= 2
        # if np.abs(new_m_d_eta) > np.abs(M_d_eta):
        #     eta_step /= 2
        if si % 100 == 0:
            theta_step, phi_step, eta_step = [step / 2 for step in [theta_step, phi_step, eta_step]]
        M_d_theta = new_m_d_theta
        M_d_phi = new_m_d_phi
        M_d_eta = new_m_d_eta
        if opt == 'min':
            theta -= theta_step * M_d_theta
            phi -= phi_step * M_d_phi
            eta -= eta_step * M_d_eta
        else:
            theta += theta_step * M_d_theta
            phi += phi_step * M_d_phi
            eta += eta_step * M_d_eta
        M_new = get_gs_stab_renyi_2(L, H, option='basis', basis=[theta, phi, eta])
        if M_new > M:
            dbg = 1
        if max(np.abs(M_d_theta), np.abs(M_d_phi), np.abs(M_d_eta)) < accuracy:
            break
        M = M_new
        # print(np.real(np.round(M, 5)), [np.real(np.round(param, 5)) for param in [M_d_theta, theta_step, M_d_phi, phi_step, M_d_eta, eta_step]])
    return M, [theta, phi, eta]



L = 5
model = 'tfim'
dirname = 'results/magic/'
lambdas = [np.round(0.2 * i, 10) for i in range(1, 10)]
j1_comp = np.zeros(len(lambdas))
j1_opt = np.zeros(len(lambdas))
jm1_comp = np.zeros(len(lambdas))
jm1_opt = np.zeros(len(lambdas))
for li in range(len(lambdas)):
    filename = dirname + 'basis_test_' + model + '_L_' + str(L) + '_lambda_' + str(lambdas[li])
    if not path.exists(filename):
        H1 = get_H(L, lambdas[li], 1, model='mim')
        comp_M_1 = get_gs_stab_renyi_2(L, H1, option='comp')
        opt_M_fraustrated, opt_basis_fraustrated = gradient_descent(H1, L)
        Hm1 = get_H(L, lambdas[li], -1, model='mim')
        comp_M_m1 = get_gs_stab_renyi_2(L, Hm1, option='comp')
        opt_M_unfr, opt_basis_unfr = gradient_descent(Hm1, L, opt='min')
        pickle.dump([comp_M_1, opt_M_fraustrated, opt_basis_fraustrated,
                     comp_M_m1, opt_M_unfr, opt_basis_unfr], open(filename, 'wb'))
    else:
        [comp_M_1, opt_M_fraustrated, opt_basis_fraustrated,
         comp_M_m1, opt_M_unfr, opt_basis_unfr] = pickle.load(open(filename, 'rb'))
    j1_comp[li] = comp_M_1
    j1_opt[li] = opt_M_fraustrated
    jm1_comp[li] = comp_M_m1
    jm1_opt[li] = opt_M_unfr

    print(li, j1_comp[li], j1_opt[li], jm1_comp[li], jm1_opt[li])
# plt.plot(lambdas, j1_comp)
# plt.plot(lambdas, j1_opt, '--')
# plt.show()
# plt.plot(lambdas, jm1_comp)
# plt.plot(lambdas, jm1_opt, '--')
# plt.show()
#
