import random
import matplotlib.pyplot as plt
import scipy.sparse as sps
from test import avg_time_elapsed
from Meeting01.Copy_of_ReadInIASI import readIASIfun
from algorithms.pcg import *
from algorithms.pcg_cholesky import facto_cholesky_incomplete


# sub-problem
def extract_sub_matrix(matrix, rows, cols):
    result = matrix[np.ix_(rows, cols)]
    return result


CorrSub, CovSub, sigmaSub = readIASIfun()

A = sps.csr_matrix(CorrSub)

print('\nAvg Time Elapsed of solving the system using sub-matrix\n')
# try to plot sub-matrix size against computational time.
basic_cg = []
diagonal = []
cholesky = []
ridge = []

# threshold for finding sub-matrix / compute directly
# how the hyper-parameters varied

for size in range(5, A.shape[0], 20):
    print('attempt!!!!!!!!!!!!!!!!')
    print(size)
    row = random.sample(range(0, A.shape[0]-1), size)
    col = random.sample(range(0, A.shape[0]-1), size)
    subA = sps.csr_matrix(extract_sub_matrix(CorrSub, rows=row, cols=col))
    b = np.ones(size)
    np.random.seed(1)
    x0 = np.random.rand(size)
    m, _, k = avg_time_elapsed(ConjugateGradient, 10, subA, x0, b)
    basic_cg.append(m/k)
    # diagonal matrix
    D = sps.diags(subA.diagonal(), format='csr')
    m, _, k = avg_time_elapsed(PreconditionedConjugateGradient, 10, subA, x0, b, D)
    diagonal.append(m/k)
    # cholesky
    T = sps.csr_matrix(facto_cholesky_incomplete(subA))
    m, _, k = avg_time_elapsed(PreconditionedConjugateGradient, 10, subA, x0, b, T)
    cholesky.append(m/k)
    # ridge
    gamma = np.delete(np.linspace(0, 0.2, 100), 0)
    record = []
    for g in gamma:
        RR = sps.csr_matrix(g * np.identity(size))
        m, sd, k = avg_time_elapsed(PreconditionedConjugateGradient, 1, subA, x0, b, RR)
        record.append(m / k)

    g = gamma[np.argmin(record)]
    RR = sps.csr_matrix(g * np.identity(size))
    m, sd, k = avg_time_elapsed(PreconditionedConjugateGradient, 10, subA, x0, b, RR)
    ridge.append(m/k)

print(basic_cg)
print(diagonal)
print(cholesky)
print(ridge)

'''
plt.plot(range(5, A.shape[0], 10), basic_cg, label='cg')
plt.plot(range(5, A.shape[0], 10), diagonal, label='diagonal')
plt.plot(range(5, A.shape[0], 10), cholesky, label='cholesky')
plt.plot(range(5, A.shape[0], 10), ridge, label='ridge')
plt.legend()
plt.show()
'''