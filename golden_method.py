import math
goldenRation = (1 + sqrt(5))/2
def f(x):
    return x**2 - 3
def golden_method(a, b, acc):
    while abs(b - a) > acc:
        x1 = b - (b - a)/goldenRatio
        x2 = a + (b - a)/goldenRatio
        if f(x1) <= f(x2):
            a = x1
        else:
            b = x2
    return a
