import numpy as np


class PlanarGame:
    """
    Parameters:
    * S : The edge set of the graph for the game, as tuples of vertices
    * (n, m) : The size of the lattice that the graph embeds into for the game
    """

    def __init__(
        self, S: list, n: int, m: int, directed: bool = True, torus: bool = False
    ):
        # Alice's input set
        self.S = S
        # Bob's input set
        self.T = S
        self.n = n
        self.m = m

        # Uniform probability matrix for each input pair
        self.prob_mat = np.ones(shape=(len(self.S), len(self.T))) / (
            len(self.S) * len(self.T)
        )
        self.A = self.line_segments(torus=torus)
        # Bob's output set
        self.B = self.A
        self.pred_mat = self.value_matrix(directed=directed, torus=torus)

    """
    Parameters:
    * n : the number of lattice points in the first dimension
    * m : the number of lattice points in the second dimension

    Returns: 
    * A list containing all line segments, as tuples of tuples of coordinates of their endpoints
    """

    def line_segments(self, torus=False):
        LX, LY = np.meshgrid(np.arange(self.n), np.arange(self.m))
        L = np.vstack((LX.flatten(), LY.flatten())).T
        A = []
        for i in range(L.shape[0]):
            for j in range(L.shape[0]):
                if i != j:
                    p_1 = L[i]
                    p_2 = L[j]
                    if not torus:
                        A.append([p_1, p_2])
                    if torus:
                        A.append(([p_1, p_2], "interior"))
                        A.append(([p_1, p_2], "wrap"))
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

    @staticmethod
    def consistent(edge_a: tuple, edge_b: tuple, line_a: tuple, line_b: tuple):
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

    @staticmethod
    def cross(line_a: tuple, line_b: tuple):
        """
        Helper function checking whether p_2 lies on the line segment from p_1 to p_3

        Parameters:
        * p_1, p_2, p_3: Three collinear points

        Returns:
        * True if p_2 lies on line segment from p_1 to p_3
        """

        def on_segment(p_1, p_2, p_3):
            if (
                p_2[0] <= max(p_1[0], p_3[0])
                and p_2[0] >= min(p_1[0], p_3[0])
                and p_2[1] <= max(p_1[1], p_3[1])
                and p_2[1] >= min(p_1[1], p_3[1])
            ):
                return True

        """
        Parameters:
        * p_1, p_2, p_3: Three points
        Returns: 
        * 0 if p_1, p_2, and p_3 are collinear
        * 1 if p_1, p_2, p_3 are oriented clockwise 
        * 2 if p_1, p_2, p_3 are oriented counterclockwise
        """

        def orientation(p_1, p_2, p_3):
            o = (p_2[1] - p_1[1]) * (p_3[0] - p_2[0]) - (p_2[0] - p_1[0]) * (
                p_3[1] - p_2[1]
            )
            if o == 0:
                return 0
            if o > 0:
                return 1
            if o < 0:
                return 2

        p_0 = tuple(line_a[0])
        p_1 = tuple(line_a[1])
        q_0 = tuple(line_b[0])
        q_1 = tuple(line_b[1])

        o_1 = orientation(p_0, p_1, q_0)
        o_2 = orientation(p_0, p_1, q_1)
        o_3 = orientation(q_0, q_1, p_0)
        o_4 = orientation(q_0, q_1, p_1)

        # If all four vertices are distinct, check if they cross anywhere
        if len({p_0, p_1, q_0, q_1}) == 4:
            # General case
            if o_1 != o_2 and o_3 != o_4:
                return True

            # Collinear cases
            if o_1 == 0 and on_segment(p_0, q_0, p_1):
                return True
            if o_2 == 0 and on_segment(p_0, q_1, p_1):
                return True
            if o_3 == 0 and on_segment(q_0, p_0, q_1):
                return True
            if o_4 == 0 and on_segment(q_0, p_1, q_1):
                return True
            return False

        # If the two edges share exactly one vertex, check if they cross away from the shared vertex
        if len({p_0, p_1, q_0, q_1}) == 3:
            # find the shared vertex
            line_a_set = {p_0, p_1}
            line_b_set = {q_0, q_1}
            (p_shared,) = line_a_set & line_b_set

            # find the two distinct vertices
            line_a_set.remove(p_shared)
            line_b_set.remove(p_shared)
            (p_0,) = line_a_set
            (p_1,) = line_b_set

            o = orientation(p_0, p_shared, p_1)
            if o == 0:  # if collinear, check if p_shared lies between p_0 and p_1
                if on_segment(p_0, p_shared, p_1):
                    return False
                else:
                    return True
            else:
                return False

        # If the two edges are the same edge, do not consider them to cross
        else:
            return False

    """
    Returns: 
    * The matrix V corresponding to the value function for the (G, n, m)-planarity game, where V[i, j, k, l] is 1 if (A[i], B[j]) is a winning answer to the inputs (S[k], T[l]), and 0 otherwise
    """

    def value_matrix(self, directed=True, torus=False):
        V_mat = np.ones(shape=(len(self.A), len(self.B), len(self.S), len(self.T)))
        for a in range(len(self.A)):
            for b in range(len(self.B)):
                for s in range(len(self.S)):
                    for t in range(len(self.T)):
                        edge_a = self.S[s]
                        edge_b = self.T[t]
                        line_a = self.A[a]
                        line_b = self.B[b]

                        if directed and not torus:
                            # Winning condition 1
                            # Alice and Bob must return the same point exactly on the same vertices
                            if not PlanarGame.consistent(
                                edge_a, edge_b, line_a, line_b
                            ):
                                V_mat[a, b, s, t] = 0

                            # Planarity
                            if PlanarGame.cross(line_a, line_b):
                                V_mat[a, b, s, t] = 0
                        if not directed and not torus:
                            v_0 = edge_a[0]
                            v_1 = edge_a[1]
                            w_0 = edge_b[0]
                            w_1 = edge_b[1]
                            p_0 = tuple(line_a[0])
                            p_1 = tuple(line_a[1])
                            q_0 = tuple(line_b[0])
                            q_1 = tuple(line_b[1])

                            # Alice and Bob must return the same point exactly on the same vertices
                            if len({v_0, v_1, w_0, w_1}) != len({p_0, p_1, q_0, q_1}):
                                V_mat[a, b, s, t] = 0

                            if PlanarGame.cross(line_a, line_b):
                                V_mat[a, b, s, t] = 0

        return V_mat


# S = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
# n = 1
# m = 3
S = eval(S)

planar_game = PlanarGame(S=S, n=n, m=m, directed=True)
prob = planar_game.prob_mat
pred = planar_game.pred_mat
