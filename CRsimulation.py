import numpy as np
import matplotlib.pyplot as plt


def u_we(u_max, f, T, N):
    def u(t):
        return u_max * np.sin(2 * np.pi * f * t)

    return u(np.arange(0, N * T, T))


u_max, f, T, N = 1, 100,  0.000015, 2000
R = 1000
C = 10e-7
uwe = u_we(u_max, f, T, N)
uwy = np.zeros(N)

for i in range(1, N):
    uwy[i] = (uwe[i] - uwe[i - 1] + uwy[i - 1]) / (T * (1.0 / (R * C) + 1.0 / T))

plt.xlabel('t')
plt.ylabel('U')
plt.plot(np.arange(0, N * T, T), uwy, 'k', color='blue')
plt.plot(np.arange(0, N * T, T), uwe, 'k', color='red')
plt.show()
print(max(uwy)/max(uwe))


def static_characteristic(f):
    def u_we(u_max, f, T, N):
        def u(t):
            return u_max * np.sin(2 * np.pi * f * t)

        return u(np.arange(0, N * T, T))

    u_we_max, T, N = 1, 0.000015, 2000
    R = 1000
    C = 10e-7
    uwe = u_we(u_max, f, T, N)
    uwy = np.zeros(N)

    for i in range(1, N):
        uwy[i] = (uwe[i] - uwe[i - 1] + uwy[i - 1]) / (T * (1.0 / (R * C) + 1.0 / T))

    u_wy_max = max(uwy)
    return u_wy_max / u_we_max


freq = np.linspace(0.00001, 1000, 200)
plt.xlabel('frequency [Hz]')
plt.ylabel('Uwy/Uwe')
plt.plot(freq[:len(freq)-2], list(map(static_characteristic, freq[:len(freq)-2])), 'k', color='blue')
plt.annotate(f'uwy/uwe(1/2piRC) is {static_characteristic(1 / (2.0*np.pi* R*C))}',
             (200, 0.4))
plt.show()
