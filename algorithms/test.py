import scipy.sparse as sps

from Meeting01.Copy_of_ReadInIASI import readIASIfun
from algorithms.pcg import *
from algorithms.pcg_cholesky import facto_cholesky_incomplete

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


print('\nAvg Time Elapsed of computing the preconditioner\n')
# diagonal preconditioner
start = process_time()
D = sps.diags(A.diagonal(), format='csr')
end = process_time() - start
print('\n Time for compute diagonal is: {}\n'.format(end))


# preconditioner for incomplete cholesky
start = process_time()
T = sps.csr_matrix(facto_cholesky_incomplete(A))
end = process_time() - start
print('\n Time for compute cholesky is: {}\n'.format(end))


# Preconditioner for ridge regression
start = process_time()
gamma = np.arange(0.001, 0.2, 0.001)
record = []
for g in gamma:
    RR = sps.csr_matrix(g * np.identity(n))
    m, sd = avg_time_elapsed(PreconditionedConjugateGradient, 10, A, x0, b, RR)
    record.append(m)

g = np.min(record)
RR = sps.csr_matrix(g * np.identity(n))
end = process_time() - start
print('\n Time for compute ridge is: {}\n'.format(end))


print('\nAvg Time Elapsed of solving the system\n')
# Basic cg method
m, sd = avg_time_elapsed(ConjugateGradient, 10, A, x0, b)
print('CG   {:.2e} ± {:.2e}'.format(m, sd))


# Try using diagonal as preconditioner
m, sd = avg_time_elapsed(PreconditionedConjugateGradient, 10, A, x0, b, D)
print('PCG diagonal {:.2e} ± {:.2e}'.format(m, sd))


# Try using incomplete cholesky as preconditioner
m, sd = avg_time_elapsed(PreconditionedConjugateGradient, 10, A, x0, b, T)
print('PCG cholesky {:.2e} ± {:.2e}'.format(m, sd))


# Try ridge regression
m, sd = avg_time_elapsed(PreconditionedConjugateGradient, 10, A, x0, b, RR)
print('PCG ridge with gamma={} {:.2e} ± {:.2e}'.format(g, m, sd))
