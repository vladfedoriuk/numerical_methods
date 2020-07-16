from numpy import dot
import numpy


def Av(A, v):
    return numpy.dot(A, v)


def conjGrad(A, Av, x, b, tol=1.0e-9):
    n = len(b)
    r = b - Av(A, x)
    s = r.copy()
    for i in range(n):
        u = Av(A, s)
        alpha = dot(s.transpose(), r) / dot(s.transpose(), u)
        x = x + alpha * s
        r = b - Av(A, x)
        if (numpy.sqrt(dot(r.transpose(), r))) < tol:
            break
        else:
            beta = -dot(r.transpose(), u) / dot(s.transpose(), u)
            s = r + beta * s
    return x, i


A = numpy.array([4, -1, 0, -1, 4, -1, 0, -1, 4])

A = A.reshape(3, 3)
b = numpy.array([2, 6, 2])
b = b.reshape(3, 1)
x, i = conjGrad(A, Av, numpy.array([0, 0, 0]).reshape(3, 1), b)

print(x)
