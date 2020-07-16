import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse

n = 1000
h = 0.01
a = np.array([-1 / (h ** 2)] * n)
b = np.array([1 + 2 / (h ** 2)] * (n + 1))
c = np.array([-1 / (h ** 2)] * n)
b[0] = 1
c[0] = 0
b[n] = 2
a[n - 1] = -1
A = scipy.sparse.diags([a, b, c], [-1, 0, 1], shape=(n + 1, n + 1)).toarray()
A[A.shape[0] - 1, 0] = -1
print("A\n", A)


def gauss_seidel(A, b, n):
    x = np.zeros(A.shape[0])
    L = np.tril(A)
    U = A - L
    for i in range(n):
        x = b - np.dot(U, x)
        x = np.linalg.solve(L, x)
    return x


def jacobi(A, b, N):
    x = np.zeros(A.shape[0])
    D = scipy.sparse.diags([A.diagonal()], [0], shape=(n + 1, n + 1)).toarray()
    A = A - D
    for i in range(N):
        x = -np.matmul(A, x) + b
        x = np.linalg.solve(D, x)
    return x


def sor(A, b, N):
    x = np.zeros(A.shape[0])
    w = 1.7
    for i in range(1, N):
        for k in range(0, len(x)):
            x[k] = (1 - w) * x[k] + w * (b[k] - np.dot(A[k, 0:k], x[0:k]) - np.dot(A[k, k + 1:], x[k + 1:])) / A[k, k]
    return x


def relax(A, b, N):
    x = np.ones(A.shape[0])
    a = 1
    for i in range(N):
        r = b - np.dot(A, x)
        x = x + (np.dot(r, np.dot(A, r)) / np.dot(np.dot(A, r), np.dot(A, r))) * (b - np.dot(A, x))
    return x


def jacobi_iter(A, b, N):
    x = np.zeros(A.shape[0])
    for i in range(1, N):
        for k in range(0, len(x)):
            x[k] = (b[k] - np.dot(A[k, 0:k], x[0:k]) - np.dot(A[k, k + 1:], x[k + 1:])) / A[k, k]
    return x


f = np.zeros(n + 1)
f[0] = 1
r = np.arange(0, n + 1) * h

"""plt.title("gauss_seidel")
x = gauss_seidel(A, f, 10)
plt.scatter(r, x)
plt.show()

plt.title("np.linalg.solve()")
x = np.linalg.solve(A, f)
plt.scatter(r, x)
plt.show()

plt.title("jacobi method")
x = jacobi_iter(A, f, 10000)
plt.scatter(r, x)
plt.show()
"""
plt.title("SOR")
x = sor(A, f, 10000)
plt.scatter(r, x)
plt.show()
"""
plt.title("Richardson")
x = relax(A, f, 10)
plt.scatter(r, x)
plt.show()
"""
