#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import numpy.linalg as ln
import scipy as sp
from math import sqrt


# Objective function
def f(x):
    return x[0] ** 3 + 8 * x[1] ** 3 - 6 * x[0] * x[1] + 5


# Gradient
def jac(x):
    return np.array([3 * x[0] ** 2 - 6 * x[1], 24 * x[1] ** 2 - 6 * x[0]])


# Hessian
def hess(x):
    return np.array([[6 * x[0], -6], [-6, 48 * x[1]]])


def dogleg_method(Hk, gk, Bk, trust_radius):
    """Dogleg trust region algorithm.

                  / tau . pU            0 <= tau <= 1,
        p(tau) = <
                  \ pU + (tau - 1)(pB - pU),    1 <= tau <= 2.

    where:

        - tau is in [0, 2]
        - pU is the unconstrained minimiser along the steepest descent direction.
        - pB is the full step.

    pU is defined by the formula::

                gT.g
        pU = - ------ g
               gT.B.g

    and pB by the formula::

        pB = - B^(-1).g

    If the full step is within the trust region it is taken.  
    Otherwise the point at which the dogleg trajectory intersects the trust region is taken.  
    This point can be found by solving the scalar quadratic equation:

        ||pU + (tau - 1)(pB - pU)||^2 = delta^2
    """

    # Compute the Newton point.
    # This is the optimum for the quadratic model function.
    # If it is inside the trust radius then return this point.
    pB = -np.dot(Hk, gk)
    norm_pB = sqrt(np.dot(pB, pB))

    # Test if the full step is within the trust region.
    if norm_pB <= trust_radius:
        return pB

    # Compute the Cauchy point.
    # This is the predicted optimum along the direction of steepest descent.
    pU = - (np.dot(gk, gk) / np.dot(gk, np.dot(Bk, gk))) * gk
    dot_pU = np.dot(pU, pU)
    norm_pU = sqrt(dot_pU)

    # If the Cauchy point is outside the trust region,
    # then return the point where the path intersects the boundary.
    if norm_pU >= trust_radius:
        return trust_radius * pU / norm_pU

    # Find the solution to the scalar quadratic equation.
    # Compute the intersection of the trust region boundary
    # and the line segment connecting the Cauchy and Newton points.
    # This requires solving a quadratic equation.
    # ||p_u + tau*(p_b - p_u)||**2 == trust_radius**2
    # Solve this for positive time t using the quadratic formula.
    pB_pU = pB - pU
    dot_pB_pU = np.dot(pB_pU, pB_pU)
    dot_pU_pB_pU = np.dot(pU, pB_pU)
    fact = dot_pU_pB_pU ** 2 - dot_pB_pU * (dot_pU - trust_radius ** 2)
    tau = (-dot_pU_pB_pU + sqrt(fact)) / dot_pB_pU

    # Decide on which part of the trajectory to take.
    return pU + tau * pB_pU


def trust_region_dogleg(func, jac, hess, x0, initial_trust_radius=1.0,
                        max_trust_radius=100.0, eta=0.15, gtol=1e-4,
                        maxiter=100):
    """An algorithm for trust region radius selection.
    
        First calculate rho using the formula::

                    f(xk) - f(xk + pk)
            rho  =  ------------------,
                      mk(0) - mk(pk)

        where the numerator is called the actual reduction and the denominator is the predicted reduction.  
        Secondly choose the trust region radius for the next iteration.  Finally decide if xk+1 should be shifted to xk.
    """
    xk = x0
    trust_radius = initial_trust_radius
    k = 0
    while True:

        gk = jac(xk)
        Bk = hess(xk)
        Hk = np.linalg.inv(Bk)

        pk = dogleg_method(Hk, gk, Bk, trust_radius)

        # Actual reduction.
        act_red = func(xk) - func(xk + pk)

        # Predicted reduction.
        pred_red = -(np.dot(gk, pk) + 0.5 * np.dot(pk, np.dot(Bk, pk)))

        # Rho.
        rhok = act_red / pred_red
        if pred_red == 0.0:
            rhok = 1e99
        else:
            rhok = act_red / pred_red

        # Calculate the Euclidean norm of pk.
        norm_pk = sqrt(np.dot(pk, pk))

        # Rho is close to zero or negative, therefore the trust region is shrunk.
        if rhok < 0.25:
            trust_radius = 0.25 * norm_pk
        else:
            # Rho is close to one and pk has reached the boundary of the trust region, therefore the trust region is expanded.
            if rhok > 0.75 and norm_pk == trust_radius:
                trust_radius = min(2.0 * trust_radius, max_trust_radius)
            else:
                trust_radius = trust_radius

        # Choose the position for the next iteration.
        if rhok > eta:
            xk = xk + pk
        else:
            xk = xk

        # Check if the gradient is small enough to stop
        if ln.norm(gk) < gtol:
            break

        # Check if we have looked at enough iterations
        if k >= maxiter:
            break
        k = k + 1
    return xk


result = trust_region_dogleg(f, jac, hess, [5, 5])
print("Result of trust region dogleg method: {}".format(result))
print("Value of function at a point: {}".format(f(result)))
