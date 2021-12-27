import numpy as np
import quaternion
import tensornetwork as tn
import basicOperations as bops

def myconj_1(q):
    return -np.conjugate(q * np.quaternion(0, 0, 1, 0))


def myconj_2(q):
    return -q * np.quaternion(0, 0, 0, 1)


def myconj_3(q):
    return -np.conjugate(q * np.quaternion(0, 1, 0, 0))


def random_q():
    r = np.random.randint(2)
    arr = np.zeros(4)
    arr[r] = 1
    return np.quaternion(int(0 == r), int(1 == r), int(2 == r), int(3 == r)) * (np.random.randint(2) * 2 - 1)
    # return np.quaternion(np.random.randint(2) - 0.5,
    #                      np.random.randint(2) - 0.5,
    #                      np.random.randint(2) - 0.5,
    #                      np.random.randint(2) - 0.5)


def round(arr, digits):
    shape = arr.shape



steps = 10000
epsilon = np.zeros((2, 2, 2, 2, 2, 2, 2, 2), dtype='quaternion')
v4 = np.zeros((2, 2, 2, 2), dtype='quaternion')
for step in range(steps):
    vec = np.array([random_q(), random_q()])
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    v4[i, j, k, l] += vec[i] * vec[j] * vec[k] * vec[l] / steps
                    for ip in range(2):
                        for jp in range(2):
                            for kp in range(2):
                                for lp in range(2):
                                    if [i, j, k, l] == [0, 0, 0, 0] and [ip, jp, kp, lp] == [0, 0, 0, 1]:
                                        b = 1
                                    if [i, j, k, l] == [0, 0, 0, 1] and [ip, jp, kp, lp] == [0, 0, 0, 0]:
                                        b = 1
                                    epsilon[i, j, k, l, ip, jp, kp, lp] += \
                                        vec[i] * vec[j] * vec[k] * vec[l] * np.conj(vec[ip] * vec[jp] * vec[kp] * vec[lp]) / steps
                                        # vec[i] * myconj_1(vec[j]) * myconj_2(vec[k]) * myconj_3(vec[l]) * \
                                        # np.conjugate(vec[ip] * myconj_1(vec[jp]) * myconj_2(vec[kp]) * myconj_3(vec[lp])) / steps
print(epsilon.reshape(16, 16))
