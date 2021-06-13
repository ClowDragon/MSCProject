import scipy.sparse as sps

from Meeting01.Copy_of_ReadInIASI import readIASIfun
from algorithms.pcg import *

CorrSub, CovSub, sigmaSub = readIASIfun()

A = sps.csr_matrix(CorrSub)
n = A.shape[0]
b = np.ones(n)
np.random.seed(1)
x0 = np.random.rand(n)


def avg_time_elapsed(method, iterations, *argv):
    elapsed_time = np.empty(iterations)
    for i in range(iterations):
        _, _, _, t = method(*argv)
        elapsed_time[i] = t
    return np.mean(elapsed_time), np.std(elapsed_time)


def standard_solver(A, x0, b):
    linear_solver = np.linalg.solve if isinstance(A, np.ndarray) else spsolve
    start = process_time()
    x = linear_solver(A, b)
    return x, None, None, process_time() - start


print('\nAvg Time Elapsed\n')

m, sd = avg_time_elapsed(ConjugateGradient, 10, A, x0, b)
print('CG   {:.2e} ± {:.2e}'.format(m, sd))

D = sps.diags(A.diagonal(), format='csr')
m, sd = avg_time_elapsed(PreconditionedConjugateGradient, 10, A, x0, b, D)
print('PCGd {:.2e} ± {:.2e}'.format(m, sd))