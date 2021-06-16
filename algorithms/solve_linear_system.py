from time import process_time

import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve_triangular


def Jacobi(A, x0, b, tol=1.e-10, max_iter=200):
    start = process_time()
    n = A.shape[0]
    x = x0
    d = 1. / A.diagonal()
    DA = A.copy()
    if isinstance(DA, np.ndarray):
        np.fill_diagonal(DA, 0.)
    else:
        DA.setdiag(0.)
    bnorm = np.linalg.norm(b)
    res = [np.linalg.norm(A.dot(x) - b) / bnorm]
    k = 0
    while res[-1] > tol and k < max_iter:
        x = d * (b - DA.dot(x))
        res.append(np.linalg.norm(A.dot(x) - b) / bnorm)
        k += 1
    return x, k, np.array(res), process_time() - start


def GaussSeidel(A, x0, b, tol=1.e-10, max_iter=200):
    start = process_time()
    n = A.shape[0]
    x = x0
    if isinstance(A, np.ndarray):
        MD, N = np.tril(A), np.triu(A, k=1)
        linear_solver = np.linalg.solve
    else:
        MD, N = sps.tril(A, format='csr'), sps.triu(A, k=1, format='csr')
        linear_solver = spsolve_triangular
    bnorm = np.linalg.norm(b)
    res = [np.linalg.norm(A.dot(x) - b) / bnorm]
    k = 0
    while res[-1] > tol and k < max_iter:
        x = linear_solver(MD, b - N.dot(x))
        res.append(np.linalg.norm(A.dot(x) - b) / bnorm)
        k += 1
    return x, k, np.array(res), process_time() - start


def GaussSeidelCC(A, x0, b, tol=1.e-10, max_iter=200):
    start = process_time()
    n = A.shape[0]
    x = x0
    d = A.diagonal()
    if isinstance(A, np.ndarray):
        M, N = np.tril(A, k=-1), np.triu(A, k=1)
    else:
        M, N = sps.tril(A, k=-1, format='csr'), sps.triu(A, k=1, format='csr')
    bnorm = np.linalg.norm(b)
    res = [np.linalg.norm(A.dot(x) - b) / bnorm]
    k = 0
    while res[-1] > tol and k < max_iter:
        v = N.dot(x)
        xk = np.zeros(n)
        for i in range(n):
            s = M[i, :].dot(xk)
            xk[i] = (b[i] - v[i] - s) / d[i]
        x = xk
        res.append(np.linalg.norm(A.dot(x) - b) / bnorm)
        k += 1
    return x, k, np.array(res), process_time() - start


def SteepestDescent(A, x0, b, tol=1.e-10, max_iter=200):
    start = process_time()
    r = b - A.dot(x0)
    x = x0
    bnorm = np.linalg.norm(b)
    res = [np.linalg.norm(r) / bnorm]
    k = 0
    while res[-1] > tol and k < max_iter:
        Ar = A.dot(r)
        alpha = (r @ r) / (r @ Ar)
        x = x + alpha * r
        r = r - alpha * Ar
        res.append(np.linalg.norm(r) / bnorm)
        k += 1
    return x, k, np.array(res), process_time() - start