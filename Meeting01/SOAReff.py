# Generated with SMOP  0.41
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs
from scipy.linalg import toeplitz


# Define a function to construct a SOAR matrix:

def SOAReff(n=100, L=0.5, a=1, maxval=None, multval=None, ridgeval=None):
    # Variables:
    # n=number of points on the circle - I have been using 100 and 200
    # theta=360/n; angle between each of the points
    # L=lengthscale - typical values range between 0.1 and 0.5
    # a=radius of circle -- good default value is 1
    theta = 2 * np.pi / maxval

    # circle for the value of n specified
    # 2*pi/n*(abs(i-l))
    v = np.zeros(n)
    # SOAReff.m:15
    vtemp = np.zeros(maxval)
    # SOAReff.m:16
    for l in range(maxval):
        # calculates the 'great circle' distance between the two points we
        # are looking at
        thetaj = theta * abs(1 - l)
        # vtemp(l)=multval*((1+abs(2*a*sin(thetaj/2))/L)*exp(-abs(2*a*sin(thetaj/2))/L));
        vtemp[l] = multval * (1 + abs(2 * a * np.sin(thetaj / 2)) / L) * \
                   np.exp(-abs(2 * a * np.sin(thetaj / 2)) / L)
    # SOAReff.m:22
    vtemp[0] = vtemp[0] + ridgeval
    # SOAReff.m:24
    v[:maxval] = vtemp
    # SOAReff.m:25
    v = [v[0]] + vtemp[::-1][:-1].tolist()
    # SOAReff.m:26
    # C = gallery('circul',v);
    C = toeplitz([v[0]] + v[::-1][:-1], v)
    # SOAReff.m:29
    if C[0, 1] - C[1, 0] != 0:
        C = C + C.conj()
    # SOAReff.m:31

    cval, _ = eigs(C, 1, which='SR')
    # SOAReff.m:33
    if cval < 0:
        C = C + (abs(cval) + 0.5 * np.random.rand()) * np.eye(n)
    # SOAReff.m:35

    return sparse.csr_matrix(C)


if __name__ == '__main__':
    SOAReff(3, 0.3, 1, 3, 1, 0)
