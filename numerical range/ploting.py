import matplotlib.pyplot as plt
import numpy as np
import math


# 1 2+i 3 4 2+i 2-i 4 0 3 4 i -i 4 0 -i 1

def max_eigen(A):
    eigen_vals, vectors = np.linalg.eig(A)
    max_eig = eigen_vals[0]
    max_vec = vectors[:, 0]
    for i in range(0, np.shape(eigen_vals)[0]):
        if eigen_vals[i] > max_eig:
            max_vec = vectors[:, i]
            max_eig = eigen_vals[i]
    return max_eig, max_vec


iterations = 100000
N = 4

print("Input coefficients")
l = input().replace("i", "j").split()
A = np.array([complex(x) for x in l])
A = np.reshape(A, (4, 4))
print(A)

x = np.zeros((iterations, 1))
y = np.zeros((iterations, 1))

# rozklad gaussa

for k in range(iterations):
    v_re = np.random.normal(loc=0.0, scale=1.0, size=4)
    v_im = np.random.normal(loc=0.0, scale=1.0, size=4) * 1j;
    v = v_re + v_im
    vH = np.conjugate(v)
    vH = np.transpose(vH)
    value = np.matmul(vH, A)
    value = np.dot(value, v) / (np.dot(vH, v))
    x[k, 0] = value.real
    y[k, 0] = value.imag

plt.xlabel("Re(x)")
plt.ylabel("Im(x)")
plt.title("numerical range")
plt.scatter(x, y, color="blue", alpha=0.5, s=[1])

AH = np.conjugate(A)
AH = np.transpose(AH)
Aah = (A - AH) / 2
Ah = (A + AH) / 2
print("Ah", Ah)
print("Aah", Aah)

max_e, max_v = max_eigen(Ah)
max_vH = np.conjugate(max_v)
max_v = np.transpose(max_v)
right = np.matmul(max_vH, Aah).dot(max_v) + max_e
plt.scatter(right.real, right.imag, color="yellow")

points = np.arange(0, 2 * np.pi, 0.01)
for i in points:
    Bah = Aah * math.sin(i) / 1j
    Bh = Ah * math.cos(i)
    max_be, max_v = max_eigen(Bh + Bah)
    max_v = max_v / np.linalg.norm(max_v)
    max_vH = np.conjugate(max_v)
    max_v = np.transpose(max_v)
    right = (np.matmul(max_vH, A).dot(max_v))
    plt.scatter(right.real, right.imag, alpha=0.5, s=[0.5], color="red")

plt.show()
