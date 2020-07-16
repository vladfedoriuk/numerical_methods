import numpy as np
import matplotlib.pyplot as plt


def write(y):
    global file, n
    file.write("{} {}\n".format(n * h, y))
    print("{} {}\n".format(n * h, y))
    n += 1


n = 1000
h = 0.01
y = np.zeros((n + 1))

y[0] = 1
y[n] = 0
y[1] = 4 / (6 - h ** 2)
y[2] = 3 - 16 / (6 - h ** 2)

for i in range(2, n - 1):
    y[i + 1] = -y[i - 1] - (h**2 - 2)*y[i]

file = open("n4.txt", "w")
n = 0
vec2 = np.vectorize(write)
vec2(y)

file.close()
n = np.arange(0, 1001) * h
plt.scatter(n, y)
plt.show()
