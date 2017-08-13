###############################################################################
# trust-region dogleg algorithm                                               #
# Copyright (C) 2017 FUNNYDMAN                                                #
#                                                                             #
#                                                                             #
# This program is free software: you can redistribute it and/or modify        #
# it under the terms of the GNU General Public License as published by        #
# the Free Software Foundation, either version 3 of the License, or           #
# (at your option) any later version.                                         #
#                                                                             #
# This program is distributed in the hope that it will be useful,             #
# but WITHOUT ANY WARRANTY; without even the implied warranty of              #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
# GNU General Public License for more details.                                #
#                                                                             #
# You should have received a copy of the GNU General Public License           #
# along with this program.  If not, see <http://www.gnu.org/licenses/>.       #
#                                                                             #
###############################################################################
#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import numpy.linalg as ln
import scipy as sp
from math import sqrt


def f(x):
    return x[0]**3 + 8*x[1]**3 - 6*x[0]*x[1] + 5

# Derivative

def jac(x):
    return np.array([3*x[0]**2 - 6*x[1], 24*x[1]**2 - 6*x[0]])

def hess(x):
    return np.array([[6*x[0], -6], [-6, 48*x[1]]])
    


    
def dogleg_method(Hk, gk, Bk, trust_radius):

    # Calculate the full step and its norm.
    pB = -np.dot(Hk, gk)
    norm_pB = sqrt(np.dot(pB, pB))

    # Test if the full step is within the trust region.
    if norm_pB <= trust_radius:
        return pB

    # Calculate pU.
    pU = - (np.dot(gk, gk) / np.dot(gk, np.dot(Bk, gk))) * gk
    dot_pU = np.dot(pU, pU)
    norm_pU = sqrt(dot_pU)

    # Test if the step pU exits the trust region.
    if norm_pU >= trust_radius:
        return trust_radius * pU / norm_pU

    # Find the solution to the scalar quadratic equation.

    pB_pU = pB - pU
    dot_pB_pU = np.dot(pB_pU, pB_pU)
    dot_pU_pB_pU = np.dot(pU, pB_pU)
    fact = dot_pU_pB_pU**2 - dot_pB_pU * (dot_pU - trust_radius**2)
    tau = (-dot_pU_pB_pU + sqrt(fact)) / dot_pB_pU
    
    # Decide on which part of the trajectory to take.
    return pU + tau * pB_pU
    

def trust_region_dogleg(func, jac, hess, x0, initial_trust_radius=1.0,
                        max_trust_radius=100.0, eta=0.15, 
                        maxiter=100):
    xk = x0
    trust_radius = initial_trust_radius
    
    for k in range(maxiter):
        
        gk = jac(xk)
        Bk = hess(xk)
        Hk = np.linalg.inv(Bk)
        
        pk = dogleg_method(Hk, gk, Bk, trust_radius)
       
        act_red = func(xk) - func(xk + pk)

        pred_red = -(np.dot(gk, pk) + 0.5 * np.dot(pk, np.dot(Bk, pk)))
        
        rhok = act_red / pred_red
        if pred_red == 0.0:
            rhok = 1e99
        else:
            rhok = act_red / pred_red
        norm_pk = sqrt(np.dot(pk, pk))
        
        
        if rhok < 0.25:
            trust_radius = 0.25 * norm_pk
        else: 
            if rhok > 0.75 and norm_pk == trust_radius:
                trust_radius = min(2.0*trust_radius, max_trust_radius)
            else:
                trust_radius = trust_radius
        
        
        if rhok > eta:
            xk = xk + pk
        else:
            xk = xk
    return xk
        
result = trust_region_dogleg(f, jac, hess, [5, 5])
print("Result of trust region dogleg method:")
print(result)
print("Value of function at a point:")
print(f(result))
