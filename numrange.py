import matplotlib.pyplot as plt
import numpy as np
import math


def shoelace_formula(x, y, absoluteValue=True):
    result = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    if absoluteValue:
        return abs(result)
    else:
        return result


def max_eigen(A):
    eigen_vals, vectors = np.linalg.eig(A)
    max_eig = eigen_vals[0]
    max_vec = vectors[:, 0]
    for i in range(0, np.shape(eigen_vals)[0]):
        if eigen_vals[i] > max_eig:
            max_vec = vectors[:, i]
            max_eig = eigen_vals[i]
    return max_eig, max_vec


class NumRange:
    def __init__(self, A=None):
        self.A = A
        self.AH = np.conjugate(self.A)
        self.AH = np.transpose(self.AH)
        self.Aah = (self.A - self.AH) / 2
        self.Ah = (self.A + self.AH) / 2

    def __str__(self):
        return self.A.__str__()

    def most_right_point(self):
        max_e, max_v = max_eigen(self.Ah)
        max_vH = np.conjugate(max_v)
        max_v = np.transpose(max_v)
        right = np.matmul(max_vH, self.Aah).dot(max_v) + max_e
        return right

    def __compute_vertical(self, alpha):
        Bah = self.Aah * math.sin(alpha) / 1j
        Bh = self.Ah * math.cos(alpha)
        max_be, max_v = max_eigen(Bh + Bah)
        max_v = max_v / np.linalg.norm(max_v)
        max_vH = np.conjugate(max_v)
        max_v = np.transpose(max_v)
        right = (np.matmul(max_vH, self.A).dot(max_v))
        return right

    def edge(self):
        points = np.arange(0, 2 * np.pi, 0.1)
        x, y = np.zeros(len(points)), np.zeros(len(points))
        for (i, j) in enumerate(points):
            right = self.__compute_vertical(j)
            x[i] = right.real
            y[i] = right.imag
        return x, y

    def __compute_value(self):
        v_re = np.random.normal(loc=0.0, scale=1.0, size=4)
        v_im = np.random.normal(loc=0.0, scale=1.0, size=4) * 1j;
        v = v_re + v_im
        vH = np.conjugate(v)
        vH = np.transpose(vH)
        value = np.matmul(vH, self.A)
        value = np.dot(value, v) / (np.dot(vH, v))
        return value

    def figure(self):
        iterations = 100000

        x = np.zeros(iterations)
        y = np.zeros(iterations)

        for k in range(iterations):
            value = self.__compute_value()
            x[k] = value.real
            y[k] = value.imag
        return x, y

    def __set_name(self):
        plt.xlabel("Re(x)")
        plt.ylabel("Im(x)")
        plt.title("numerical range")

    def plot(self, file):
        self.__set_name()

        #x, y = self.figure()
        #plt.scatter(x, y, color="blue", alpha=0.5, s=[1])

        x, y = self.edge()
        plt.fill(x, y)
        plt.scatter(x, y, alpha=0.5, s=[0.5], color="red")

        right = self.most_right_point()
        plt.scatter(right.real, right.imag, color="yellow")

        area = shoelace_formula(x, y)
        print(area)

        plt.savefig(file)
        plt.show()
