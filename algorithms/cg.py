import numpy as np


def cg(A, b, x):
    """
    A function to solve [A]{x} = {b} linear equation system with the
    conjugate gradient method.
    ========== Parameters ==========
    A : matrix
        A real symmetric positive definite matrix.
    b : vector
        The right hand side (RHS) vector of the system.
    x : vector
        The starting guess for the solution.
    """
    r = b - np.dot(A, x)
    p = r
    rsold = np.dot(np.transpose(r), r)

    for i in range(len(b)):
        Ap = np.dot(A, p)
        alpha = rsold / np.dot(np.transpose(p), Ap)
        x = x + np.dot(alpha, p)
        r = r - np.dot(alpha, Ap)
        rsnew = np.dot(np.transpose(r), r)
        if np.sqrt(rsnew) < 1e-8:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x
