import scipy.sparse as sps
from scipy.sparse.linalg import eigsh
from algorithms.pcg import *
from test import avg_time_elapsed

from Meeting01.Copy_of_ReadInIASI import readIASIfun

np.set_printoptions(suppress=True)

CorrSub, CovSub, sigmaSub = readIASIfun()

A = sps.csr_matrix(CorrSub)
n = A.shape[0]
b = np.ones(n)
np.random.seed(1)
x0 = np.random.rand(n)

# Minimum eigenvalue method : find sigma1,2 and v1 ,P = (s2 - s1) * V1 * V1 T
start = process_time()
evals_small, evecs_small = eigsh(A, 2, which='SM')

M = (evals_small[1] - evals_small[0]) * (np.dot(evecs_small[0], np.transpose(evecs_small[0])))
ME = (evals_small[1] - evals_small[0]) * (np.dot(evecs_small[0], np.transpose(evecs_small[0])))
P = sps.csr_matrix(A + ME * np.identity(n))
end = process_time() - start
print('\n Time for compute minimum eigenvalue preconditioner is: {}\n'.format(end))


m, sd, k = avg_time_elapsed(PreconditionedConjugateGradient, 10, A, x0, b, P)
print('Minimum eigenvalue method with {} iterations: {:.2e} ± {:.2e}'.format(k, m, sd))
print('Minimum eigenvalue method time per iteration :{:.2e}'.format(m / k))
