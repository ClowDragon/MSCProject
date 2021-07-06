# Generated with SMOP  0.41
import numpy as np
from scipy import sparse
from SOAReff import SOAReff

# create block R
    

def makeRICeff(p=None, pvec=None, overlap=None):

    if sum(pvec) > p:
        # check overlap works
        pass
    else:
        if sum(pvec) == p:
            # says where to start blocks
            overlap[0] = 1
    # makeRICeff.m:10
            for inc1 in range(len(pvec) - 1):
                overlap[inc1 + 1] = overlap[inc1] + pvec[inc1]
    # makeRICeff.m:12
        else:
            # error because p<sum(pvec)
            pass
    
    # generate random blocks
    R = np.zeros((p, p))
    # makeRICeff.m:19
    for inc in range(len(pvec)):
        print(pvec[inc])
        index = overlap[inc] + pvec[inc] - 1
    # speye(pvec(inc))+5*sprand(pvec(inc),pvec(inc),0.8).*SOAReff(pvec(inc),0.3,1,min(pvec(inc),200),1,0);
        R[overlap[inc]:index+1, overlap[inc]:index+1] =\
            np.eye(pvec[inc]) + 5 * sparse.rand(pvec[inc], pvec[inc], 0.8) * \
            SOAReff(pvec[inc], 0.3, 1, min(pvec[inc], 200), 1, 0)
    # makeRICeff.m:24
    # R = R+R';
    # also generate inverse?
    
    return R


if __name__ == '__main__':
    print(makeRICeff(p=100, pvec=[3,5,22,20,20,29], overlap=[0,0,0,0,0,0]))

