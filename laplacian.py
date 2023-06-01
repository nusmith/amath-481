import numpy as np
from scipy.sparse import spdiags

# Laplacian matrix
h = 0.5 # step size in space (x and Y)
n = 20
m = n*n # total size of matrix

e1 = np.ones(m) # vector of ones
Low1 = np.tile(np.concatenate((np.ones(n-1), [0])), (n,)) # Lower diagonal 1
Low2 = np.tile(np.concatenate(([1], np.zeros(n-1))), (n,)) #Lower diagonal 2
                                    # Low2 is NOT on the second lower diagonal,
                                    # it is just the next lower diagonal we see
                                    # in the matrix.

Up1 = np.roll(Low1, 1) # Shift the array for spdiags
Up2 = np.roll(Low2, n-1) # Shift the other array

L = spdiags([e1, e1, Low2, Low1, -4*e1, Up1, Up2, e1, e1],
                         [-(m-n), -n, -n+1, -1, 0, 1, n-1, n, (m-n)], m, m, format='csc')
def Laplacian(A):
    return L*A