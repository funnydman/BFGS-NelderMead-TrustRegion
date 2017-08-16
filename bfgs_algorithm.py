
'''
    Pure Python/Numpy implementation of the BFGS algorithm.
    Reference: https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm
'''

#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import numpy.linalg as ln
import scipy as sp
import scipy.optimize


# Objective function
def f(x):
    return x[0]**2 - x[0]*x[1] + x[1]**2 + 9*x[0] - 6*x[1] + 20


# Derivative
def f1(x):
    return np.array([2 * x[0] - x[1] + 9, -x[0] + 2*x[1] - 6])


def bfgs_method(f, fprime, x0, maxiter=None, epsi=10e-3):
    """
    Minimize a function func using the BFGS algorithm.
    
    Parameters
    ----------
    func : f(x)
        Function to minimise.
    x0 : ndarray
        Initial guess.
    fprime : fprime(x)
        The gradient of `func`.
    """
    
    if maxiter is None:
        maxiter = len(x0) * 200

    # initial values
    k = 0
    gfk = fprime(x0)
    N = len(x0)
    # Set the Identity matrix I.
    I = np.eye(N, dtype=int)
    Hk = I
    xk = x0
   
    while ln.norm(gfk) > epsi and k < maxiter:
        
        # pk - direction of search
        
        pk = -np.dot(Hk, gfk)
        
        # Line search constants for the Wolfe conditions.
        # Repeating the line search
        
        # line_search returns not only alpha
        # but only this value is interesting for us

        line_search = sp.optimize.line_search(f, f1, xk, pk)
        alpha_k = line_search[0]
        
        xkp1 = xk + alpha_k * pk
        sk = xkp1 - xk
        xk = xkp1
        
        gfkp1 = fprime(xkp1)
        yk = gfkp1 - gfk
        gfk = gfkp1
        
        k += 1
        
        ro = 1.0 / (np.dot(yk, sk))
        A1 = I - ro * sk[:, np.newaxis] * yk[np.newaxis, :]
        A2 = I - ro * yk[:, np.newaxis] * sk[np.newaxis, :]
        Hk = np.dot(A1, np.dot(Hk, A2)) + (ro * sk[:, np.newaxis] *
                                                 sk[np.newaxis, :])
        
    return (xk, k)


result, k = bfgs_method(f, f1, np.array([1, 1]))

print('Result of BFGS method:')
print('Final Result (best point): %s' % (result))
print('Iteration Count: %s' % (k))
