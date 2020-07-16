import numpy as np


def cQuotientIteration(A, m, max_iter):
    I = np.identity(A.shape[0])
    v = np.array([1, 1, 1])
    v = v/np.linalg.norm(v)
    #mu = np.dot(v, np.dot(A, v))
    mu = m
    for t in range(max_iter):
        try:
            v = np.linalg.solve(A - mu * I, v)
        except:
            #print("Matrix is singular")
            return v, mu

        v /= np.linalg.norm(v)
        mu = np.dot(v, np.dot(A, v))

    return v, mu


def checkDiagonal(arr):
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if i == j:
                continue
            else:
                if abs(arr[i][j]) > 0.000001:
                    return False
    return True


vectors = np.identity(3)


def qrFactorization(arr):
    temp = arr
    i = 0
    while True:
        global vectors
        Q, R = np.linalg.qr(temp)
        vectors = np.matmul(vectors, Q)
        temp = np.dot(R, Q)
        if checkDiagonal(temp):
            print("Number of Factorizations: " + str(i + 1))
            break
        else:
            i += 1
    return temp


def printLambda(arr):
    count = 1
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if i == j:
                temp = arr[i][j]
                if abs(temp) < 0.000001:
                    temp = 0
                print("Lamda" + str(count) + ": " + str(temp))
                count += 1


def eigen_value(A, v):
    return np.matmul(v.transpose(), A).dot(v)


def power_method(A):
    v = np.ones(A.shape[1])
    v /= np.linalg.norm(v)
    ev = eigen_value(A, v)
    m_prev = 0
    while True:
        Av = A.dot(v)
        v_new = Av/np.linalg.norm(Av)

        ev_new = eigen_value(A, v_new)
        if np.abs(ev - ev_new) < 0.000001:
            break

        if np.abs(ev - ev_new) < 20:
            w, m  = cQuotientIteration(A, ev_new, 10)
            if abs(m_prev - m) > 0.0000001:
                print("Rayleigh value: ", m)
                print("Rayleigh vector: ", w)
                m_prev = m

        v = v_new
        ev = ev_new
    return ev_new, v_new


def find_all_power(A):
    print("All using power method:")
    vals = []
    vecs = []
    Matrix = A
    for i in range(0, A.shape[1]):
        e, v = power_method(Matrix)
        print(e, v)
        vals.append(e)
        vecs.append(v)
        Matrix = Matrix - np.outer(v, v)*e
    return vals, vecs


A = np.array([1, 2, 3, 2, 4, 5, 3, 5, -1])
A = A.reshape(3, 3)

#v, mu = cQuotientIteration(A, 1, 10)
#print("v=", v)
#print("mu=", mu)

printLambda(qrFactorization(A))
print(vectors)

e, v = np.linalg.eig(A)

print(e)
print(v)

#print(power_method(A))
find_all_power(A)
# print(v, mu)
