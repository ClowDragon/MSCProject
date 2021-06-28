import matplotlib.pyplot as plt
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
    number_of_iterations = np.empty(iterations)
    for i in range(iterations):
        _, iteration, _, t = method(*argv)
        elapsed_time[i] = t
        number_of_iterations[i] = iteration
    return np.mean(elapsed_time), np.std(elapsed_time), np.mean(number_of_iterations)


# Consider both time & time/iteration
if __name__ == '__main__':
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
    # gamma value for sub-matrix?
    start = process_time()
    gamma = np.arange(0.001, 0.2, 0.001)
    record = []
    for g in gamma:
        RR = sps.csr_matrix(g * np.identity(n))
        m, sd, k = avg_time_elapsed(PreconditionedConjugateGradient, 1, A, x0, b, RR)
        record.append(m / k)

    g = gamma[np.argmin(record)]
    RR = sps.csr_matrix(g * np.identity(n))
    end = process_time() - start
    print('\n Time for compute ridge is: {}\n'.format(end))
    plt.plot(gamma, record)
    plt.title('gamma against time per iteration')
    plt.xlabel('gamma value')
    plt.ylabel('time taken per iteration')
    plt.show()

    print('\nAvg Time Elapsed of solving the system using full matrix\n')
    # Basic cg method
    m, sd, k = avg_time_elapsed(ConjugateGradient, 10, A, x0, b)
    print('CG with {} iterations:  {:.2e} ± {:.2e}'.format(k, m, sd))
    print('CG time per iteration :{:.2e}'.format(m / k))

    # Try using diagonal as preconditioner
    m, sd, k = avg_time_elapsed(PreconditionedConjugateGradient, 10, A, x0, b, D)
    print('PCG diagonal with {} iterations: {:.2e} ± {:.2e}'.format(k, m, sd))
    print('PCG diagonal time per iteration :{:.2e}'.format(m / k))

    # Try using incomplete cholesky as preconditioner
    m, sd, k = avg_time_elapsed(PreconditionedConjugateGradient, 10, A, x0, b, T)
    print('PCG cholesky with {} iterations: {:.2e} ± {:.2e}'.format(k, m, sd))
    print('PCG cholesky time per iteration :{:.2e}'.format(m / k))

    # Try ridge regression
    m, sd, k = avg_time_elapsed(PreconditionedConjugateGradient, 10, A, x0, b, RR)
    print('PCG ridge with gamma={:.2e}, {} iterations: {:.2e} ± {:.2e}'.format(g, k, m, sd))
    print('PCG ridge time per iteration :{:.2e}'.format(m / k))

