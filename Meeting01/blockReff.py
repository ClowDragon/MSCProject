# Generated with SMOP  0.41
from scipy import sparse
from scipy.sparse.linalg import eigs
from smop.libsmop import *

from makeRICeff import makeRICeff


# blockReff.m


@function
def blockReff(pvec=None, pdist=None, pcorr=None, *args, **kwargs):
    varargin = blockReff.varargin
    nargin = blockReff.nargin

    # Create a block R matrix
    # pvec - vector with the size of the blocks (sum == p)
    # pdistribution - select correlation function for each block (currently only
    # random is option but could extend to spatial rather than pseudo IC)
    # pcorr - strength of correlation in each block

    # Ordering of blocks: from left to right/top to bottom of above diagonal blocks
    # Then symmetrise the matrix

    pnum = length(pvec)
    # blockReff.m:14
    p = sum(pvec)
    # blockReff.m:15
    pstart = ones(concat([1, pnum + 1]))
    # blockReff.m:16
    pstart[arange(2, end())] = np.cumsum(pvec(arange(1, end()))) + 1
    # blockReff.m:17
    # generate block diagonals
    R = dot(2, makeRICeff(p, pvec))
    # blockReff.m:19
    # Rtemp = (R+R');
    # Make off diagonal blocks
    val = 1
    # blockReff.m:22
    for inc in arange(1, pnum - 1).reshape(-1):
        for inc2 in arange(inc + 1, pnum).reshape(-1):
            if pdist(val) == 1:
                R[arange(pstart(inc), pstart(inc + 1) - 1), arange(pstart(inc2), pstart(inc2 + 1) - 1)] = \
                    dot(pcorr(val), sparse.rand(pvec(inc), pvec(inc2), 0.3))
            # blockReff.m:26
            else:
                if pdist(val) == 2:
                    pass
            val = val + 1
    # blockReff.m:31

    # make symmetric
    R = R + R.T
    # blockReff.m:37
    m = eigs(R, 1, which='SR')
    # blockReff.m:38
    # check PD and if not bump up diagonal
    if m <= 0:
        R = R + dot((abs(m) + 0.01), sparse.eye(p))
    # blockReff.m:41

    return R, m


if __name__ == '__main__':
    pass
