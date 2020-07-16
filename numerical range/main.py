from numrange import NumRange
import numpy as np
from productnumrange import ProductNumRange
import sys

"""print("Input coefficients")
l = input().replace("i", "j").split()
A = np.array([complex(x) for x in l])
"""
file = sys.argv[1]
l = sys.argv[2:]


def f(x):
    x = x.replace("i", "j")
    x = complex(x)
    return x


A = list(map(f, l))
A = np.reshape(A, (4, 4))
nrange = NumRange(A)
nrange.plot(file)

productnrange = ProductNumRange(A)
productnrange.plot(file)
