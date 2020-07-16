import numpy as np
import matplotlib.pyplot as plt


def lu_decomposition(c, d, e):
    n = len(d)
    for k in range(1, n):
        l = c[k - 1] / d[k - 1]
        d[k] = d[k] - l * e[k - 1]
        c[k - 1] = l
    return c, d, e


def lu_solve(c, d, e, b):
    n = len(d)
    for k in range(1, n):
        b[k] = b[k] - c[k - 1] * b[k - 1]
    b[n - 1] = b[n - 1] / d[n - 1]
    for k in range(n - 2, -1, -1):
        b[k] = (b[k] - e[k] * b[k + 1]) / d[k]
    return b


n = int(input("Input Matrix Size"))
h = 1 / n
b = [0] * (n + 1)
c = [0] * n
a = [0] * n
vec = np.vectorize(lambda x, y: x + y)
b = list(vec(b, -2 / (h ** 2)))
a = list(vec(a, 1 / (h ** 2)))
c = list(vec(c, 1 / (h ** 2)))
f = [0] * (n + 1)
b[0] = b[n] = f[n] = 1
c[0] = a[n-1] = 0

a, b, c = lu_decomposition(a, b, c)
f = lu_solve(a, b, c, f)

n = np.arange(0, n+1) * h
plt.scatter(n, f)
plt.show()

file = open("N2.txt", "w")
n = 0
for i in range(0, len(f)):
    file.write("{} {}\n".format(n * h, f[i]))
    n += 1
file.close()
