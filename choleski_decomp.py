from numpy import dot
from numpy import sqrt
from numpy import array

def choleski(a):
    n = len(a)
    for k in range(n):
        try:
            a[k, k] = sqrt(a[k, k] - dot(a[k, 0:k], a[k, 0:k]))
        except ArithmeticError:
            print("Matrix is not positive definite")
        for i in range(k+1, n):
            a[i, k] = (a[i, k] - dot(a[i, 0:k], a[k, 0:k]))/a[k, k]
    for k in range(1, n):
        a[0:k, k] = 0.0
    return a

def choleski_solve(L, b):
    n = len(b)
    #Solution for [L]{y} = {b}
    for k in range(n):
        b[k] = (b[k] - dot(L[k, 0:k], b[0:k]))/L[k, k]
    #Solution for [L_transpose]{x} = {y}
    for k in range(n-1, -1, -1):
        b[k] = (b[k] - dot(L[k+1:n, k], b[k+1:n]))/L[k, k]
    return b


N = int(input("Input a matrix size: "))

print("Input coefficients")
inf = list(map(float, input().split()))
a = array(inf).reshape(N, N)
print("A=", a)

print("Input a constant vector")
b = list(map(float, input().split()))
print(b)


L = choleski(a)
print("L=", L)
x = choleski_solve(L, b)
print("Solution:", x)