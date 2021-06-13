import numpy as np


def calc_ti(A, T, i):
    """
    calc_ti(...)
    returns the value of tii of the dense Cholesky factorization
    Parameters
    ----------
    A : positive definite square matrix
    T : the result of dense Cholesky factorization
    i : positive integer between 0 and the dimension of A-1
    Returns
    -------
    out : the value of tii
    """

    if i == 0:
        return np.sqrt(A[i, i])
    return np.sqrt(A[i, i] - sum(T[i, :i] ** 2))


def calc_tji(A, T, i, j):
    """
    calc_tji(...)
    returns the value of tji of the dense Cholesky factorization
    Parameters
    ----------
    A : positive definite square matrix
    T : the result of dense Cholesky factorization
    i : positive integer between 0 and the dimension of A-1
    j : positive integer between 0 and i
    Returns
    -------
    out : the value of tji
    """
    return (A[i, j] - sum(T[i, :i] * T[j, :i])) / T[i, i]


def facto_cholesky_incomplete_REC(A, T, i):
    """
    facto_cholesky_incomplete_REC(...)
    completes T with the value of tii and tji for j from 0 to i-1 of the incomplete Cholesky
    factorization and returns a call of itself for next index
    Parameters
    ----------
    A : positive definite square matrix
    T : the result of incomplete Cholesky factorization
    i : positive integer between 0 and the dimension of A-1
    Returns
    -------
    out : itself with a call on next index
    """

    if i == len(A):
        return T
    T[i, i] = calc_ti(A, T, i)

    for j in range(i + 1, len(A)):
        if A[j, i] != 0:
            T[j, i] = calc_tji(A, T, i, j)
    return facto_cholesky_incomplete_REC(A, T, i + 1)


def facto_cholesky_incomplete(A):
    """
    facto_cholesky_incomplete(...)
    returns the matrix of the incomplete Cholesky factorization
    Parameters
    ----------
    A : positive definite square matrix
    Returns
    -------
    out : the matrix of the incomplete Cholesky factorization
    """

    T = np.zeros([len(A), len(A)])
    return facto_cholesky_incomplete_REC(A, T, 0)


def triangular_solve_down(A, b):
    """
    triangular_solve_down(...)
    returns the solution to the equation Ax = b where A is a lower triangular matrix
    Parameters
    ----------
    A : a lower triangular matrix
    b : a column vector solution of the equation Ax = b
    Returns
    -------
    out : The vector x, solution to the equation Ax = b
    """

    n = len(A)

    x = np.zeros((n, 1))

    for ligne in range(n):
        somme = b[ligne][0]
        for colonne in range(ligne):
            somme -= A[ligne][colonne] * x[colonne]
        x[ligne] = somme / A[ligne][ligne]
    return x


def triangular_solve_up(A, b):
    """
    triangular_solve_up(...)
    returns the solution to the equation Ax = b where A is a upper triangular matrix
    Parameters
    ----------
    A : a upper triangular matrix
    b : a column vector solution of the equation Ax = b
    Returns
    -------
    out : The vector x, solution to the equation Ax = b
    Time complexity
    ---------------
    O(n**2)
    Space complexity
    ----------------
    O(n)
    """
    n = len(A)

    x = np.zeros((n, 1))

    for ligne in range(n - 1, -1, -1):
        somme = b[ligne][0]
        for colonne in range(n - 1, ligne, -1):
            somme -= A[ligne][colonne] * x[colonne]
        x[ligne] = somme / A[ligne][ligne]
    return x


def solve(T, Tt, r):
    """
    solve(...)
    returns the solution x to the equation T*Tt*x = b
    Parameters
    ----------
    T : a lower triangular matrix
    Tt : a upper triangular matrix
    b : vector solution of the equation T*Tt*x = b
    x : initial estimation of the searched vector
    Returns
    -------
    out : The vector x, solution to the equation T*Tt*x = b
    """
    y = triangular_solve_down(T, r)
    z = triangular_solve_up(Tt, y)
    return z


def pcg_cholesky(A, b, x):
    """
    conjugate_gradient_preconditioned(...)
    returns the solution to the equation Ax = b
    Parameters
    ----------
    A : positive definite square matrix
    b : vector solution of the equation Ax = b
    x : initial estimation of the searched vector
    Returns
    -------
    out : The vector x, solution to the equation Ax = b
    Time complexity
    ---------------
    inferior to n**3
    Space complexity
    ----------------
    O(n**2)
    """
    r = b - A.dot(x)  # The inverse of the gradient of f
    T = facto_cholesky_incomplete(A)  # Incomplete Cholesky preconditionner
    Tt = T.transpose()

    z = solve(T, Tt, r)
    p = z  # First direction of the being built base
    rsold = (r.transpose()).dot(z)

    for i in range(1, 10 ** 6):
        Ap = A.dot(p)
        alpha = rsold / np.dot(np.transpose(p), Ap)  # Coordinate of the solution in the base
        x = x + alpha * p  # Variable that converges towards the solution
        r = r - alpha * Ap  # New gradient
        z = solve(T, Tt, r)
        rsnew = np.dot(np.transpose(r), z)
        if np.sqrt(rsnew[0][0]) < 10 ** (-10):
            break

        p = z + (rsnew / rsold) * p  # New direction
        rsold = rsnew
    return x
