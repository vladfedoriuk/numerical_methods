import numpy as np


def gauss_elimin(a, b):
    n = len(b)
    for k in range(0, n - 1):
        for i in range(k + 1, n):
            if a[i, k] != 0.0:
                lam = a[i, k] / a[k, k]
                a[i, k + 1: n] = a[i, k + 1:n] - lam * a[k, k + 1:n]
                b[i] = b[i] - lam * b[k]
    for k in range(n - 1, -1, -1):
        b[k] = (b[k] - np.dot(a[k, k + 1: n], b[k + 1: n])) / a[k, k]
    return b


N = int(input("Input a matrix size: "))

print("Input coefficients")
inf = list(map(float, input().split()))
a = np.array(inf).reshape(N, N)
print("A=", a)

print("Input a constant vector")
b = list(map(float, input().split()))
print(b)

print("Solution:", gauss_elimin(a, b))



