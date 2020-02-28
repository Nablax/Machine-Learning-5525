import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# generate random A and b
def generate_rd_A_b(shape_A, shape_b):
    A = np.random.random(shape_A)
    b = np.random.random(shape_b)
    return A, b


# use SVD to find the pseudoinverse
def lsq(A, b):
    U, sigma, VT = np.linalg.svd(A, full_matrices=False) # full_matrices=False we get sigma a square matrix
    V, UT, sigma_inv_diag = np.transpose(VT), np.transpose(U), np.diag(1/sigma)
    A_pinv = (V.dot(sigma_inv_diag)).dot(UT)
    w = A_pinv.dot(b)
    return w


# iteration approach
def lsq_iter(A, b):
    w_hat = lsq(A, b)
    mu = 1/np.square(np.linalg.norm(A))
    iter_t = 500
    w = 0
    E_lst = []
    for i in range(iter_t):
        w = w - mu * np.transpose(A).dot(A.dot(w) - b)
        E = np.linalg.norm(w - w_hat)
        E_lst.append(E)
    plt.plot(E_lst)
    plt.xlabel('iteration times')
    plt.ylabel('||w - w_hat||')
    plt.show()
    return w


if __name__ == '__main__':
    row_size = 20
    col_A = 10
    col_b = 1
    shape_A = np.array((row_size, col_A))
    shape_b = np.array((row_size, col_b))
    A, b = generate_rd_A_b(shape_A, shape_b)
    w1 = lsq(A, b)
    w2 = lsq_iter(A, b)
