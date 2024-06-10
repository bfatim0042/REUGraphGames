import numpy as np
from toqito.nonlocal_games.nonlocal_game import NonlocalGame


class PlanarGame(NonlocalGame):

    def __init__(self, S, n, m):
        # Alice's input set
        self.S = S
        # Bob's input set
        self.T = S
        self.n = n
        self.m = m
        # Alice's output set
        self.A = self.line_segments(n, m)
        # Bob's output set
        self.B = self.A
        # Uniform probability matrix for each input pair
        prob_mat = np.ones(shape=(len(self.S), len(self.T))) / (
            len(self.S) * len(self.T)
        )
        pred_mat = self.game_values()
        super().__init__(prob_mat, pred_mat)

    """
    Parameters:
    * n : the number of lattice points in the first dimension
    * m : the number of lattice points in the second dimension

    Returns: 
    * A list containing all line segments, as tuples of tuples of coordinates of their endpoints
    """

    def line_segments(self, n, m):
        LX, LY = np.meshgrid(np.arange(self.n), np.arange(self.m))
        L = np.vstack((LX.flatten(), LY.flatten())).T
        A = []
        for i in range(L.shape[0]):
            for j in range(L.shape[0]):
                if i != j:
                    p1 = L[i]
                    p2 = L[j]
                    A.append([p1, p2])
        return A

    """
    Helper function that checks if Alice's and Bob's answers are consistent with the edges they are provided
    
    Parameters:
    * edge_a : A tuple of vertices provided to Alice
    * edge_b : A tuple of vertices provided to Bob
    * line_a : The line segment Alice returns, as a tuple of points
    * line_b : The line segment Bob returns, as a tuple of points

    Returns:
    * False if Alice and Bob return different coordinates on the same vertex, or the same coordinates on different vertices
    * True otherwise 
    """

    def consistent(self, edge_a, edge_b, line_a, line_b):
        for i in range(2):
            for j in range(2):
                # Alice and Bob must map the same vertices to the same point
                if edge_a[i] == edge_b[j] and np.any(line_a[i] != line_b[j]):
                    return False
                # Alice and Bob must map different vertices to different points
                elif edge_a[i] != edge_b[j] and np.all(line_a[i] == line_b[j]):
                    return False
        return True

    """
    Helper function that checks if two line segments intersect
    https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
    
    Parameters:
    * line_a : The first line segment, as a tuple of points
    * line_b : The second line segment, as a tuple of points

    Returns:
    * True if the line segments intersect, including the case where an endpoint of one line segment lies on the other line segment
    * False otherwise
    """

    def cross(self, line_a, line_b):
        """
        Helper function checking whether p2 lies on the line segment from p1 to p3

        Parameters:
        * p1, p2, p3: Three collinear points

        Returns:
        * True if p2 lies on line segment from p1 to p3
        """

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

    def game_values(self):
        V_mat = np.ones(shape=(len(self.A), len(self.B), len(self.S), len(self.T)))
        for a in range(len(self.A)):
            for b in range(len(self.B)):
                for s in range(len(self.S)):
                    for t in range(len(self.T)):
                        edge_a = self.S[s]
                        edge_b = self.T[t]
                        line_a = self.A[a]
                        line_b = self.B[b]

                        # Winning condition 1
                        # Check if Alice and Bob's answers are the same for the same inputs
                        if not self.consistent(edge_a, edge_b, line_a, line_b):
                            V_mat[a, b, s, t] = 0

                        # Winning condition 2
                        # If all vertices are distinct, the line segments cannot cross
                        if (
                            edge_a[0] != edge_b[0]
                            and edge_a[0] != edge_b[1]
                            and edge_a[1] != edge_b[0]
                            and edge_a[1] != edge_b[1]
                        ):
                            if self.cross(edge_a, edge_b, line_a, line_b):
                                V_mat[a, b, s, t] = 0
        return V_mat


# S = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
S = [(1, 2), (1, 3), (2, 3)]
n = 1
m = 2

# prob_mat = np.ones(shape=(len(S), len(T))) / (len(S) * len(T))
planar_game = PlanarGame(S=S, n=n, m=m)
print(f"{planar_game.classical_value()=}")
