import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.optimize

iterations = 10000
N = 4


def shoelace_formula(x, y, absoluteValue=True):
    result = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    if absoluteValue:
        return abs(result)
    else:
        return result


def compute_real(params, *args):
    return -compute(params, args).real


def compute(params, *args):
    def v_2(alpha, beta):
        return np.array([np.cos(alpha), np.sin(alpha) * np.exp(1j * beta)])

    def prod(alpha_v, beta_v, alfa_w, beta_w):
        return np.kron(v_2(alpha_v, beta_v), v_2(alfa_w, beta_w))

    x = prod(params[0], params[1], params[2], params[3])
    xH = np.conjugate(x)
    xH = np.transpose(xH)
    return np.dot((np.matmul(xH, args[0])), x) / (np.dot(xH, x))


def gen_vec(siz):
    v_re = np.random.normal(loc=0.0, scale=1.0, size=siz)
    v_im = np.random.normal(loc=0.0, scale=1.0, size=siz) * 1j;
    vec = v_re + v_im
    return vec / np.dot(vec, vec)


print("Input coefficients")
l = input().replace("i", "j").split()
A = np.array([complex(x) for x in l])
A = np.reshape(A, (4, 4))
print(A)

x = np.zeros((iterations, 1))
y = np.zeros((iterations, 1))

for k in range(iterations):
    v = gen_vec(2)
    w = gen_vec(2)
    vec = np.kron(v, w)
    vecH = np.conjugate(vec)
    vecH = np.transpose(vecH)
    value = np.matmul(vecH, A)
    value = np.dot(value, vec) / (np.dot(vecH, vec))
    x[k, 0] = value.real
    y[k, 0] = value.imag

plt.xlabel("Re(x)")
plt.ylabel("Im(x)")
plt.title("product numerical range")
plt.scatter(x, y, color="blue", alpha=0.5, s=[1])

right = scipy.optimize.minimize(compute_real, np.array([0, 0, 0, 0]), A).x
p = compute(right, A)
plt.scatter(p.real, p.imag, color="yellow")

AH = np.conjugate(A)
AH = np.transpose(AH)
Aah = (A - AH) / 2
Ah = (A + AH) / 2

points = np.arange(0, 2 * np.pi, 0.1)
verticals = list()
for i in points:
    Bah = Aah * math.sin(i) / 1j
    Bh = Ah * math.cos(i)
    # right = scipy.optimize.minimize(compute_real, np.array([0, 0, 0, 0]), Bah + Bh).x
    vectors = [np.random.normal(loc=0.0, scale=1.0, size=4) for _ in range(10)]
    vectors = [scipy.optimize.minimize(compute_real, v, Bah + Bh).x for v in vectors]
    right = min(vectors, key=lambda x: compute_real(x, Bah + Bh))
    p = compute(right, A)
    plt.scatter(p.real, p.imag, alpha=0.5, s=[0.5], color="red")
    verticals.append((p.real, p.imag))

x_y = list(zip(*verticals))
area = shoelace_formula(np.array(x_y[0]), np.array(x_y[1]))
print(area)
plt.fill(np.array(x_y[0]), np.array(x_y[1]))
plt.show()

# 1 2+i 3 4 2+i 2-i 4 7i 3 4 i -i 4 0 -i 1
