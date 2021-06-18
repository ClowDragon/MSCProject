from time import process_time

import numpy as np
from scipy.sparse.linalg import spsolve


def ConjugateGradient(A, x0, b, tol=1.e-10, max_iter=2000):
    start = process_time()
    r = b - A.dot(x0)
    p = r
    x = x0
    bnorm = np.linalg.norm(b)
    res = [np.linalg.norm(r) / bnorm]
    k = 0
    while res[-1] > tol and k < max_iter:
        Ap = A.dot(p)
        pAp = p @ Ap
        alpha = (p @ r) / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        beta = (r @ Ap) / pAp
        p = r + beta * p
        res.append(np.linalg.norm(r) / bnorm)
        k += 1
    return x, k, np.array(res), process_time() - start


def PreconditionedConjugateGradient(A, x0, b, P, tol=1.e-10, max_iter=3000):
    start = process_time()
    n = A.shape[0]
    r = b - A.dot(x0)
    p = r
    x = x0
    bnorm = np.linalg.norm(b)
    res = [np.linalg.norm(r) / bnorm]
    k = 0
    while res[-1] > tol and k < max_iter:
        Ap = A.dot(p)
        pAp = p @ Ap
        alpha = (p @ r) / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        z = spsolve(P, r)
        beta = (z @ Ap) / pAp
        p = z + beta * p
        res.append(np.linalg.norm(r) / bnorm)
        k += 1
    # k is the number of iterations, res record the residual norm
    return x, k, np.array(res), process_time() - start
