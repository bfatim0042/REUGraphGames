import numpy as np
from toqito.nonlocal_games.nonlocal_game import NonlocalGame
import argparse


class PlanarGame(NonlocalGame):
    """
    Parameters:
    * S : The edge set of the graph for the game, as tuples of vertices
    * (n, m) : The size of the lattice that the graph embeds into for the game
    """

    def __init__(self, S: list, n: int, m: int, directed: bool = True):
        # Alice's input set
        self.S = S
        # Bob's input set
        self.T = S
        self.n = n
        self.m = m
        # Alice's output set
        self.A = self.line_segments()
        # Bob's output set
        self.B = self.A
        # Uniform probability matrix for each input pair
        prob_mat = np.ones(shape=(len(self.S), len(self.T))) / (
            len(self.S) * len(self.T)
        )
        if directed:
            pred_mat = self.value_matrix()
        else:
            pred_mat = self.value_matrix_undirected()
        super().__init__(prob_mat, pred_mat)

    """
    Parameters:
    * n : the number of lattice points in the first dimension
    * m : the number of lattice points in the second dimension

    Returns: 
    * A list containing all line segments, as tuples of tuples of coordinates of their endpoints
    """

    def line_segments(self):
        LX, LY = np.meshgrid(np.arange(self.n), np.arange(self.m))
        L = np.vstack((LX.flatten(), LY.flatten())).T
        A = []
        for i in range(L.shape[0]):
            for j in range(L.shape[0]):
                if i != j:
                    p_1 = L[i]
                    p_2 = L[j]
                    A.append([p_1, p_2])
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

    def value_matrix(self):
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
                        # Alice and Bob must return the same point exactly on the same vertices
                        if not PlanarGame.consistent(edge_a, edge_b, line_a, line_b):
                            V_mat[a, b, s, t] = 0

                        # Planarity
                        if PlanarGame.cross(line_a, line_b):
                            V_mat[a, b, s, t] = 0
        return V_mat

    def value_matrix_undirected(self):
        V_mat = np.ones(shape=(len(self.A), len(self.B), len(self.S), len(self.T)))
        for a in range(len(self.A)):
            for b in range(len(self.B)):
                for s in range(len(self.S)):
                    for t in range(len(self.T)):
                        edge_a = self.S[s]
                        edge_b = self.T[t]
                        line_a = self.A[a]
                        line_b = self.B[b]
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


"""
Print the classical value of the planar game

Parameters:
* planar_game (PlanarGame) : Planar game to calculate classical value for
* print_strategy (bool) : indicates whether to print out Alice's and Bob's classical strategies, as lists where the ith value indicates their strategy on the ith input
"""


def display_classical(planar_game, print_strategy=False):
    classical_value = planar_game.classical_value()
    print(f"Classical value: {classical_value['classical_value']}")
    if print_strategy:
        print("Alice's classical strategy:")
        for idx, s in enumerate(planar_game.S):
            print(
                f"{s}: {np.array(planar_game.A[int(classical_value['alice_strategy'][idx])]).tolist()}"
            )
        print("Bob's classical strategy:")
        for idx, s in enumerate(planar_game.T):
            print(
                f"{s}: {np.array(planar_game.B[int(classical_value['bob_strategy'][idx])]).tolist()}"
            )


"""
Print the quantum value of the planar game

Parameters:
* planar_game (PlanarGame) : Planar game to calculate quantum value for
* print_strategy (bool) : indicates whether to print out Alice's and Bob's quantum strategies, as POVMs 
"""


def display_quantum(planar_game, print_strategy=False):
    quantum_lower_bound = planar_game.quantum_value_lower_bound()
    print(f"Quantum value lower bound: {quantum_lower_bound['quantum_lower_bound']}")
    if print_strategy:
        for s_idx, s in enumerate(planar_game.S):
            print(f"{s}:")
            print("Alice's POVMs:")
            for a_idx, a in enumerate(planar_game.A):
                print(
                    f"{np.array(a).tolist()}:\n{np.array(quantum_lower_bound['alice_strategy'][s_idx, a_idx].value).round(3)}"
                )
            print("Bob's POVMs:")
            for b_idx, b in enumerate(planar_game.B):
                print(
                    f"{np.array(b).tolist()}:\n{np.array(quantum_lower_bound['bob_strategy'][s_idx, b_idx].value).round(3)}"
                )


"""
Run on local computer
"""


def small_embedding_values():
    small_S = []
    # small_S.append([(1, 2)])
    # small_S.append([(1, 2)])
    small_S.append([(1, 2), (2, 3), (3, 1)])

    # small_S.append([(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)])
    # small_S.append(
    #     [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1), (1, 3), (1, 4), (3, 5), (1, 4), (1, 5)]
    # )
    # small_S.append(
    #     [(1, 6), (1, 4), (1, 2), (2, 5), (2, 3), (3, 4), (3, 6), (4, 5), (5, 6)],
    # )
    quantum = True
    classical = True
    ns = True
    for S in small_S:
        for m, n in [(1, 3)]:  # , (1, 3), (1, 4), (2, 2)]:
            print(f"{S=}, {m=}, {n=}")
            planar_game = PlanarGame(S=S, n=n, m=m, directed=True)
            if ns:
                print(f"{planar_game.nonsignaling_value()=}")
            if quantum:
                display_quantum(planar_game, print_strategy=False)
            if classical:
                display_classical(planar_game, print_strategy=True)

            planar_game = PlanarGame(S=S, n=n, m=m, directed=True)
            if ns:
                print(f"{planar_game.nonsignaling_value()=}")
            if quantum:
                display_quantum(planar_game, print_strategy=False)
            if classical:
                display_classical(planar_game, print_strategy=True)


def cluster():
    planar_game = PlanarGame(S=eval(args.edges), n=args.n, m=args.m)
    print(f"{planar_game.S=}, {planar_game.m=}, {planar_game.n=}")
    print(f"{planar_game.nonsignaling_value()=}")
    display_quantum(
        planar_game,
        print_strategy=False,
    )
    display_classical(
        planar_game,
        print_strategy=True,
    )


parser = argparse.ArgumentParser(description="Test planar embedding game.")
parser.add_argument(
    "-m", "--m", help="Number of lattice points in first direction", type=int
)
parser.add_argument(
    "-n", "--n", help="Number of lattice points in second direction", type=int
)
parser.add_argument("-e", "--edges", help="Description of graph in terms of edge set")
args = parser.parse_args()
# If optional arguments are given, run with optional arguments.
# Otherwise run with whatever values are set in small_embedding_values.
if args.n is not None:
    cluster()
else:
    small_embedding_values()
