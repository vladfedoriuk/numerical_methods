import matplotlib.pyplot as plt

X = [x for x in range(10, 35, 5)]
Y = [2.12, 3.192, 4.255, 5.32, 6.38]

plt.xlabel('U wejściowe [V]')
plt.ylabel('U wyjściowe [V]')
plt.plot(X, Y, 'ro')
plt.plot(X, Y, 'k')
plt.show()