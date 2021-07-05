# Generated with SMOP  0.41
import numpy as np
from scipy import sparse
from SOAReff import SOAReff
# makeRICeff.m

# create block R
    

def makeRICeff(p=None, pvec=None, overlap=None):

    if sum(pvec) > p:
        # check overlap works
        pass
    else:
        if sum(pvec) == p:
            # says where to start blocks
            overlap[1]=1
# makeRICeff.m:10
            for inc1 in np.arange(1,len(pvec) - 1).reshape(-1):
                overlap[inc1 + 1]=overlap(inc1) + pvec(inc1)
# makeRICeff.m:12
        else:
            #error because p<sum(pvec)
            pass
    
    #generate random blocks
    R = np.zeros((p, p), dtype=complex)
    R[0][0] = complex(1, 1)
    R = sparse.bsr_matrix(R)
# makeRICeff.m:19
    for inc in np.arange(1,len(pvec)).reshape(-1):
        index=overlap(inc) + pvec(inc) - 1
# makeRICeff.m:23
        R[np.arange(overlap(inc),index),np.arange(overlap(inc),index)]=\
            sparse.eye(pvec(inc)) + np.multiply(np.dot(5,sparse.rand(pvec(inc),pvec(inc),0.8)),
                                             SOAReff(pvec(inc),0.3,1,min(pvec(inc),200),1,0))
# makeRICeff.m:24
    
    #R = R+R';
# also generate inverse?
    
    return R


if __name__ == '__main__':
    test = makeRICeff(p=1000)
    print(test)
