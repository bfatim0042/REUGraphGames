import numpy as np

S = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
# S = [(1, 2), (1, 3), (2, 3)]
T = S

# Create lattice containing all pairs of coordinates
k = 3
l = 3

LX, LY = np.meshgrid(np.arange(k), np.arange(l))
L = np.vstack((LX.flatten(), LY.flatten())).T

A = []
for i in range(L.shape[0]):
    for j in range(L.shape[0]):
        if i != j:
            p1 = L[i]
            p2 = L[j]
            A.append([p1, p2])
B = A

def consistent(edge_a, edge_b, line_a, line_b):
    for i in range(2):
        for j in range(2):
            # Alice and Bob map the same vertices to the same point
            if edge_a[i] == edge_b[j] and np.any(line_a[i] != line_b[j]):
                return False
            # Injectivity
            elif edge_a[i] != edge_b[j] and np.all(line_a[i] == line_b[j]):
                return False
    return True


# https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
def cross(edge_a, edge_b, line_a, line_b):
    # Checks if p2 lies on line segment from p1 to p3
    def on_segment(p1, p2, p3):
        if (
            p2[0] <= max(p1[0], p3[0])
            and p2[0] >= min(p1[0], p3[0])
            and p2[1] <= max(p1[1], p3[1])
            and p2[1] >= min(p1[1], p3[1])
        ):
            return True

    def orientation(p1, p2, p3):
        o = (p2[1] - p1[1]) * (p3[0] - p2[0]) - (p2[0] - p1[0]) * (p3[1] - p2[1])
        if o == 0:
            return 0
        if o > 0:
            return 1
        if o < 0:
            return 2

    if (
        edge_a[0] != edge_b[0]
        and edge_a[0] != edge_b[1]
        and edge_a[1] != edge_b[0]
        and edge_a[1] != edge_b[1]
    ):
        o1 = orientation(line_a[0], line_a[1], line_b[0])
        o2 = orientation(line_a[0], line_a[1], line_b[1])
        o3 = orientation(line_b[0], line_b[1], line_a[0])
        o4 = orientation(line_b[0], line_b[1], line_a[1])
        # General case
        if o1 != o2 and o3 != o4:
            return True

        # Collinear cases
        if o1 == 0 and on_segment(line_a[0], line_b[0], line_a[1]):
            return True
        if o2 == 0 and on_segment(line_a[0], line_b[1], line_a[1]):
            return True
        if o3 == 0 and on_segment(line_b[0], line_a[0], line_b[1]):
            return True
        if o4 == 0 and on_segment(line_b[0], line_a[1], line_b[1]):
            return True
        return False
    else:
        return False


V_mat = np.ones(shape=(len(A), len(B), len(S), len(T)))
for a in range(len(A)):
    for b in range(len(B)):
        for s in range(len(S)):
            for t in range(len(T)):
                edge_a = S[s]
                edge_b = T[t]
                line_a = A[a]
                line_b = B[b]

                # Winning condition 1
                # Check if Alice and Bob's answers are the same for the same inputs
                if not consistent(edge_a, edge_b, line_a, line_b):
                    V_mat[a, b, s, t] = 0

                # Winning condition 2
                # If all vertices are distinct, the line segments cannot cross
                if cross(edge_a, edge_b, line_a, line_b):
                    V_mat[a, b, s, t] = 0

print(V_mat)

from toqito.nonlocal_games.nonlocal_game import NonlocalGame

prob_mat = np.ones(shape=(len(S), len(T))) / (len(S) * len(T))
planar = NonlocalGame(prob_mat, V_mat)
print(f"{planar.classical_value()=}")
