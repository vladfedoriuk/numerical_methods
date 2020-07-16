import numpy as np


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


def write(y):
    global file, n
    file.write("{} {}\n".format(n * h, y))
    print("{} {}\n".format(n * h, y))
    n += 1


n = 1000
h = 0.01
b = [0] * (n + 1)
c = b.copy()
a = b.copy()
vec = np.vectorize(lambda x, y: x + y)

b = list(vec(b, h ** 2 - 2))
a = list(vec(a, 1))
c = list(vec(c, 1))
f = [0] * (n + 1)
f[0] = h ** 2
b[0] = 1 * (h ** 2)
c[0] = 0
a[n-1] = h**2
b[n] = -2*(h**2)

u = [0] * (n + 1)
u[n] = h
v = [0] * (n + 1)
v[0] = h


# A = A+ uv
# Az = b
# Aq = u

x1, x2, x3 = lu_decomposition(a, b, c)
g = lambda y: np.array(lu_solve(x1, x2, x3, y))
lst = [f, u]
vec1 = np.vectorize(g)
lst = list(map(g, lst))
z = np.array(lst[0])
q = np.array(lst[1])

v = np.array(v)
u = np.array(u)
w = z - (np.dot(v, z) / (1 + np.dot(v, q))) * q

file = open("n3.txt", "w")
n = 0
vec2 = np.vectorize(write)
vec2(w)

file.close()
