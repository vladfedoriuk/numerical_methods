import numpy as np
import matplotlib.pyplot as plt
import functools


def lu_decomp(c, d, e):
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


def plot(f, points):
    x = np.arange(-4.999, 5.001, 0.001,)
    plt.plot(x, list(map(f, x)), 'k')
    x_list = []
    y_list = []
    for x_p, y_p in points:
        x_list.append(x_p)
        y_list.append(y_p)

    plt.plot(x_list, y_list, 'ro')
    plt.show()


def curvatures(xData, yData):
    n = len(xData) - 1
    c = np.zeros(n)
    d = np.ones(n + 1)
    e = np.zeros(n)
    k = np.zeros(n + 1)
    c[0:n - 1] = xData[0:n - 1] - xData[1:n]
    d[1:n] = 2.0 * (xData[0:n - 1] - xData[2:n + 1])
    e[1:n] = xData[1:n] - xData[2:n + 1]
    k[1:n] = 6.0 * (yData[0:n - 1] - yData[1:n]) / (xData[0:n - 1] - xData[1:n]) - 6.0 * (
            yData[1:n] - yData[2:n + 1]) / (xData[1:n] - xData[2:n + 1])
    c, d, e = lu_decomp(c, d, e)
    return lu_solve(c, d, e, k)


def sinc_approximation(f, N):
    h = 10 / (N - 1)
    xData = list(map(lambda i: -5 + h * i, range(0, N)))

    def P(x):
        def phi(i):
            if abs(x - xData[i]) < 0.0001:
                return 1
            else:
                return np.sin((np.pi / h) * (x - xData[i])) / ((np.pi / h) * (x - xData[i]))

        coefs = np.array(list(map(phi, range(0, len(xData)))))
        yData = np.array(list(map(f, xData)))
        return np.dot(coefs, yData)

    return P


def evaluate_spline(points):
    x_y = list(zip(*points))
    xData = np.array(x_y[0])
    yData = np.array(x_y[1])

    def P(x):

        def findSegment(x):
            i_left = 0
            i_right = len(xData) - 1
            while True:
                if i_right - i_left <= 1:
                    return i_left
                i = (i_left + i_right) // 2
                if x < xData[i]:
                    i_right = i
                else:
                    i_left = i

        k = curvatures(xData, yData)
        i = findSegment(x)
        h = xData[i] - xData[i + 1]
        y = ((x - xData[i + 1]) ** 3 / h - (x - xData[i + 1]) * h) * k[i] / 6.0 - (
                (x - xData[i]) ** 3 / h - (x - xData[i]) * h) * k[i + 1] / 6.0 + (
                    yData[i] * (x - xData[i + 1]) - yData[i + 1] * (x - xData[i])) / h
        return y

    return P


def lagrange(points):
    def P(x):
        total = 0
        n = len(points)
        for i in range(n):
            xi, yi = points[i]

            def g(i, n):
                tot_mul = 1
                for j in range(n):
                    if i == j:
                        continue
                    xj, yj = points[j]
                    tot_mul *= (x - xj) / float(xi - xj)

                return tot_mul

            total += yi * g(i, n)
        return total

    return P


def f1(x):
    return 1 / (1 + x ** 2)


def f2(x):
    return np.exp(-x ** 2)


N = 10


def get_points(f, N):
    points_x = np.arange(-5, 5, 10 / N)
    points_y = list(map(f, points_x))
    points = list(map(tuple, zip(points_x, points_y)))
    return points


def get_points_xn(f, N):
    def xn(n):
        return 5 * np.cos(np.pi * n / (N - 1))

    points_x = [xn(n) for n in range(N)]
    points_y = list(map(f, points_x))
    points = list(map(tuple, zip(points_x, points_y)))
    return points


def get_points_sinc(f, N):
    h = 10 / (N - 1)
    points_x = list(map(lambda i: -5 + h * i, range(0, N)))
    points_y = list(map(f, points_x))
    points = list(map(tuple, zip(points_x, points_y)))
    return points


points = get_points(f1, N)
plot(f1, points)
P = lagrange(points)
plot(P, points)

points1 = get_points(f2, N)
plot(f2, points1)
P = lagrange(points1)
plot(P, points1)

print("xn")

points2 = get_points_xn(f1, N)
P = lagrange(points2)
plot(P, points2)
points3 = get_points_xn(f2, N)
P = lagrange(points3)
plot(P, points3)

print("cubic_spline")

P = evaluate_spline(points)
plot(P, points)
P = evaluate_spline(points1)
plot(P, points1)

print("sync approximation")
P = sinc_approximation(f1, N)
plot(P, get_points_sinc(f1, N))
