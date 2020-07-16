import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.optimize


accuracy = 0.000001
memoized_arguments = {}


def memoize(f):
    def g(x):
        if x not in memoized_arguments:
            memoized_arguments[x] = f(x)
        return memoized_arguments[x]

    return g


@memoize
def f(x):
    return np.exp(np.sin(x) ** 2)


def trapez_rule(a, b, N):
    x = np.linspace(a, b, N+1)
    y = np.exp(np.sin(x) ** 2)
    y_right = y[1:]
    y_left = y[:-1]
    dx = (b - a) / N
    T = (dx / 2) * np.sum(y_right + y_left)
    plt.plot(x, y)
    """for i in range(N):
        xs = [x[i], x[i], x[i + 1], x[i + 1]]
        ys = [0, f(x[i]), f(x[i + 1]), 0]
        plt.fill(xs, ys, 'b', edgecolor='b', alpha=0.2)
    plt.show()"""
    return T


def trapez_rule_iter(a, b):
    parts = 1
    N = 0
    prev = 0
    while True:
        N += 1
        interval = (b - a) / parts
        x = np.linspace(a, b, parts+1, endpoint=True)
        y = np.exp(np.sin(x) ** 2)
        y_right = y[1:]
        y_left = y[:-1]
        integral = np.sum(y_right + y_left) * (interval / 2)
        if abs(prev - integral) < accuracy:
            print("N=", N)
            break
        parts *= 2
        prev = integral
    return integral


def simpson_rule(f, a, b):
    N = 2
    prev = 0
    while True:
        dx = (b - a) / N
        x = np.linspace(a, b, N + 1)
        y = np.exp(np.sin(x) ** 2)
        S = dx / 3 * np.sum(y[0:-1:2] + 4 * y[1::2] + y[2::2])
        if abs(S - prev) < accuracy:
            break
        prev = S
        N *= 2
    return S


def simpson_3_8(f, a, b):
    def g(i):
        if i % 3 == 0:
            return 2 * f(a + i * interval_size)
        else:
            return 3 * f(a + i * interval_size)

    intervals = 1
    prev = 0
    while True:
        interval_size = (b - a) / intervals
        l = list(map(g, range(1, intervals)))
        s = sum(l) + f(a) + f(b)
        s = s * interval_size * 3 / 8
        if abs(s - prev) < accuracy:
            break
        prev = s
        intervals *= 4
    return s


def plot(f, a, b):
    plt.title("function and the second derivative graph")

    x = np.linspace(a, b, 1000)
    y = np.exp(np.sin(x) ** 2)
    spl = scipy.interpolate.InterpolatedUnivariateSpline(x, y)
    dfdx = spl.derivative()
    df2d2x = dfdx.derivative()
    plt.plot(x, y, 'b', linewidth=0.5, alpha=0.5)
    plt.plot(x, abs(df2d2x(x)), 'r', linewidth=2, alpha=0.5)
    a = plt.legend(['function', 'abs(second derivative)', ])
    plt.show()

    x = scipy.optimize.minimize(df2d2x, np.array([1.5])).x
    return abs(df2d2x(x))


def get_iteration_number(a, b, K2, accuracy):
    return np.int(np.ceil(np.sqrt(((b - a) ** 3) * K2 / (12 * accuracy))))


max_der = plot(f, 0, np.pi)
print("max der2", max_der)
N = get_iteration_number(0, np.pi, max_der, accuracy)
print(N)
print(trapez_rule(0, np.pi, N))
print(trapez_rule_iter(0, np.pi))
print(simpson_rule(f, 0, np.pi))
print(simpson_3_8(f, 0, np.pi))
