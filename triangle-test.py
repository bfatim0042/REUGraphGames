import numpy as np

S = [(1, 2), (1, 3), (2, 3)]
T = S

# Create lattice containing all pairs of coordinates
k = 1
l = 2

LX, LY = np.meshgrid(np.arange(k), np.arange(l))
L = np.vstack((LX.flatten(), LY.flatten())).T

A = []
for i in range(L.shape[0]):
    for j in range(L.shape[0]):
        if i != j:
            p1 = L[i]
            p2 = L[j]
            A.append([p1, p2])
A = np.array(A)
B = A

