print("""

Matrix should be like: [ d1 e1 0 0 ...    0
                         c1 d2 e2 0 ...   0
                         0  c2 d3 e3 ...  0
                         0  0 c3 d4 e4 ...0
                         . .  .  .  .  .. 0
                         . .  .  .  .  .. 0
                         0 0  0  0 cn-1 dn-1]
                         
""")


def lu_decomp(c, d, e):
    n = len(d)
    for k in range(1, n):
        l = c[k-1]/d[k-1]
        d[k] = d[k] - l*e[k-1]
        c[k-1] = l
    return c, d, e

def lu_solve(c, d, e, b):
    n = len(d)
    for k in range(1, n):
        b[k] = b[k] - c[k-1]*b[k-1]
    b[n-1] = b[n-1]/d[n-1]
    for k in range(n-2, -1, -1):
        b[k] = (b[k]-e[k]*b[k+1])/d[k]
    return b

c = input("Enter c vector")
c = c.split()
c = [float(x) for x in c]
print("c = ", c)

d = input("Enter d vector")
d = d.split()
d = [float(x) for x in d]
print("d = ", d)

e = input("Enter e vector")
e = e.split()
e = [float(x) for x in e]
print("e = ", e)

b = input("Enter b vector")
b = b.split()
b = [float(x) for x in b]
print("b = ", b)
try:
    if len(c) != len(e) or len(c) != len(d)-1:
        raise Exception
    else:
        c, d, e = lu_decomp(c, d, e)
        b = lu_solve(c, d, e, b)
        print(b)
except Exception:
    print("Bad vector length")







