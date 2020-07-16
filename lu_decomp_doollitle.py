from numpy import dot, array

def lu_decomp(a):
    n = len(a)
    for k in range(0, n-1):
        for i in range(k+1, n):
            if a[i, k] != 0.0:
                lam = a[i, k]/a[k, k]
                a[i,k+1:n] = a[i, k+1:n] - lam*a[k, k+1:n]
                a[i, k] =lam
    return a

def lu_solve(a, b):
    n = len(a)
    for k in range(1, n):
        b[k] = b[k] - dot(a[k, 0:k], b[0:k])
    for k in range(n-1, -1, -1):
        b[k] = (b[k] - dot(a[k, k+1:n], b[k+1:n]))/a[k, k]
    return b

N = int(input("Input a matrix size: "))

print("Input coefficients")
inf = list(map(float, input().split()))
a = array(inf).reshape(N, N)
print("A=", a)

print("Input a constant vector")
b = list(map(float, input().split()))
print(b)

print("Solution:", lu_solve(lu_decomp(a), b))

l = ['0', '1', '2']

#print(map(float, l)) - prints <map object at 0x0BC0C0F0>


