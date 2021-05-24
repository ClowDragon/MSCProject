import math
import cmath
import numpy as np
import scipy.linalg as LA
import matplotlib.pylab as plt

print(cmath.exp(-2*math.pi*1j*1*1/10))
print(((cmath.exp(-2*math.pi*1j*1*1/10))/cmath.sqrt(10)))
def SOARinv(n,L,a):
#Variables:
#n=number of points on the circle - I have been using 100 and 200
#theta=360/n; angle between each of the points
#L=lengthscale - typical values range between 0.1 and 0.5
#a=radius of circle -- good default value is 1

    theta=2*math.pi/n; #calculates the angle between each adjacent point on the 
    # circle for the value of n specified

    C=np.zeros((n,n));
    cvec = np.zeros(n)
    
    #SOAR is a circulant matrix (fully described by its first row)
    #Begin by constructing the first row, cvec
    for l in range(1,n+1):
        thetaj = theta*abs(1-l)
        cvec[l-1]=(1+abs(2*a*math.sin(thetaj/2))/L)*math.exp(-abs(2*a*math.sin(thetaj/2))/L);

    # Use inbuilt Python function to generate the full circulant matrix, C
    C = LA.circulant(cvec)
  
    lam=np.zeros(n,dtype=complex);
    evecs =np.zeros((n,n),dtype=complex)
    
    # This just calculates the eigenvalues using the definition for a circulant
    # matrix - lambda is the matrix of eigenvalues
    # For any nxn circulant matrix, the eigenvectors are the same, so I
    # calculate these too - evecs is the matrix of eigenvectors. Note that it
    # doesn't depend on the values in C!
    for m in range(0,n):
        for k in range(0,n):

            lam[m]=lam[m]+cvec[k]*cmath.exp(-2*math.pi*1j*m*k/n);
            evecs[m,k]=(((cmath.exp(-2*math.pi*1j*m*k/n))/cmath.sqrt(n)));
    
    
    #sometimes there are spurious trailing zero complex parts - get rid of
    #these
    evals=(np.real(lam));
    
    #Calculate the eigenvalues for the inverse of the SOAR matrix
    evalsin=np.diag((1./evals));
    #Calculate the inverse of the SOAR matrix using the eigenvalues and
    #vectors calulated above
    Cinv=np.real(evecs*evalsin*(evecs.conj()));
    return C,Cinv,evecs,evals

# Example using standard/sensible input values
C,Cinv,evecs,evals = SOARinv(100,0.1,1)

