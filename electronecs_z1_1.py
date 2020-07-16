import numpy as np
import matplotlib.pyplot as plt


class CRCircuit(object):
    def __init__(self, R, C):
        self.R = R
        self.C = C
        self.fd = 1.0 / (2 * np.pi * self.R * self.C)

    def __k(self, freq):
        return 1.0 / (1 + (self.fd / freq) ** 2) ** 0.5

    @staticmethod
    def __plot(X, Y, mode='ro'):
        plt.plot(X, Y, mode)
        plt.show()

    def plot(self, end, points, mode='u'):
        freq = np.linspace(0.00001, end, points)
        if mode == 'f':
            k = self.__k(freq)
            plt.annotate(f'  k(f0) is { self.__k(self.fd)}', (self.fd, self.__k(self.fd)))
            plt.scatter(self.fd, self.__k(self.fd))
            plt.xlabel('f[1/s]')
            plt.ylabel('Uwy/Uwe')
            self.__plot(freq, k, 'k')
        elif mode == 'a':
            theta = np.arctan(self.fd / freq)
            th_f0 = np.arctan(self.fd/self.fd)
            plt.annotate(f'theta for f0 is {th_f0}', (self.fd, th_f0))
            plt.scatter(self.fd, self.__k(self.fd))
            plt.xlabel('fd/f')
            plt.ylabel('theta')
            self.__plot(freq, theta, 'k')
        else:
            raise RuntimeError('unsupported mode')


if __name__ == '__main__':
    cr = CRCircuit(1000, 10e-6)
    cr.plot(800, 2000, mode='f')
    cr.plot(800, 2000, mode='a')

