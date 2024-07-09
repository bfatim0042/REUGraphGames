import numpy as np
from toqito.nonlocal_games.nonlocal_game import NonlocalGame
import argparse

class PlanarGame(NonlocalGame):
    """
    Parameters:
    * S : The edge set of the graph for the game, as tuples of vertices
    * (m, n) : The size of the lattice that the graph embeds into for the game
    """

    def __init__(
        self, S: list, m: int, n: int, directed: bool = True, torus: bool = False
    ):
        # Alice's input set
        self.S = S
        # Bob's input set
        self.T = S
        self.m = m
        self.n = n

        self.torus = torus
        if self.torus:
            if m != 1:
                raise Exception("The game has not been implemented for torus boundary conditions not on a 1 x n grid.")

        # Uniform probability matrix for each input pair
        self.prob_mat = np.ones(shape=(len(self.S), len(self.T))) / (
            len(self.S) * len(self.T)
        )
        self.A = self.line_segments(torus=torus)
        # Bob's output set
        self.B = self.A
        self.pred_mat = self.value_matrix(directed=directed, torus=torus)

        super().__init__(self.prob_mat, self.pred_mat)

    """
    Parameters:
    * n : the number of lattice points in the first dimension
    * m : the number of lattice points in the second dimension

    Returns: 
    * A list containing all line segments, as tuples of tuples of coordinates of their endpoints
    """

    def line_segments(self, torus: bool = False):
        LX, LY = np.meshgrid(np.arange(self.n), np.arange(self.m))
        L = np.vstack((LX.flatten(), LY.flatten())).T
        A = []
        for i in range(L.shape[0]):
            for j in range(L.shape[0]):
                if i != j:
                    p_1 = L[i]
                    p_2 = L[j]
                    if not torus:
                        A.append({'endpoints': [p_1, p_2], 'boundary_conditions': ""})
                    if torus:
                        A.append({'endpoints': [p_1, p_2], 'boundary_conditions': "interior"})
                        A.append({'endpoints': [p_1, p_2], 'boundary_conditions': "wrap"})
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
    def consistent(edge_a: tuple, edge_b: tuple, line_a: dict, line_b: dict):
        # Alice and Bob must map the exact same edge to the exact same line segments 
        if edge_a == edge_b: 
            if line_a['boundary_conditions'] != line_b['boundary_conditions']:
                return False


        # Alice and Bob must map the same vertices to the same points
        for i in range(2):
            for j in range(2):
                # Alice and Bob must map the same vertices to the same point
                if edge_a[i] == edge_b[j]:
                    if np.any(line_a['endpoints'][i] != line_b['endpoints'][j]):
                        return False
                
                # Alice and Bob must map different vertices to different points
                elif edge_a[i] != edge_b[j]:
                    if np.all(line_a['endpoints'][i] == line_b['endpoints'][j]):
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
    def cross(line_a: tuple, line_b: tuple, torus: bool = False):
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

        p_0 = tuple(line_a['endpoints'][0])
        p_1 = tuple(line_a['endpoints'][1])
        q_0 = tuple(line_b['endpoints'][0])
        q_1 = tuple(line_b['endpoints'][1])
        bc_0 = line_a['boundary_conditions']
        bc_1 = line_b['boundary_conditions']

        o_1 = orientation(p_0, p_1, q_0)
        o_2 = orientation(p_0, p_1, q_1)
        o_3 = orientation(q_0, q_1, p_0)
        o_4 = orientation(q_0, q_1, p_1)

        # If all four vertices are distinct, check if they cross anywhere
        if len({p_0, p_1, q_0, q_1}) == 4:
            if not torus:
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
            elif torus:
                # assume 1 x n for torus boundary conditions
                # considering two lines that do not cross (on the circle), there are four possibilities of where the boundaries occur
                if bc_0 == "wrap":
                    if bc_1 == "interior" and (min(p_0[0], p_1[0]) < min(q_0[0], q_1[0]) and max(p_0[0], p_1[0]) > max(q_0[0], q_1[0])):
                        return False
                if bc_1 == "wrap":
                    if bc_0 == "interior" and (min(q_0[0], q_1[0]) < min(p_0[0], p_1[0]) and max(q_0[0], q_1[0]) > max(p_0[0], p_1[0])):
                        return False 
                if bc_0 == "interior" and bc_1 == "interior":
                    if max(p_0[0], p_1[0]) < min(q_0[0], q_1[0]): 
                        return False 
                    if max(q_0[0], q_1[0]) < min(p_0[0], p_1[0]):
                        return False
                return True

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
            if o == 0:  
                # if collinear, check if p_shared lies between p_0 and p_1
                if on_segment(p_0, p_shared, p_1):
                    if not torus:
                        return False
                    # the line segments intersect if either of their boundary conditions is "wrap"
                    elif torus:
                        return "wrap" in [bc_0, bc_1] 
                else:
                    if not torus:
                        return True
                    elif torus: 
                        # Assume n x 1 grid
                        # if p_shared is not between p_0 and p_1, the further point needs to be "wrap" and the closer point needs to be "interior" to not intersect
                        if np.abs(p_0[0] - p_shared[0]) < np.abs(p_1[0] - p_shared[0]):
                            p_close = p_0
                            bc_close = bc_0 
                            bc_far = bc_1
                        else:
                            p_close = p_1 
                            bc_close = bc_1 
                            bc_far = bc_0
                        return not ((bc_close == "interior") and (bc_far == "wrap"))
            else:
                # Does not account for torus if not 1xn 
                return False

        # If the two edges are the same edge, do not consider them to cross
        elif len({p_0, p_1, q_0, q_1}) <= 2:
            return False
            
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

                        if directed:
                            # Winning condition 1
                            # Alice and Bob must return the same point exactly on the same vertices
                            if not PlanarGame.consistent(
                                edge_a, edge_b, line_a, line_b
                            ):
                                V_mat[a, b, s, t] = 0
                                
                            else:
                                # Planarity
                                if PlanarGame.cross(line_a, line_b, torus):
                                    V_mat[a, b, s, t] = 0

                        elif not directed and not torus:
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

                            if PlanarGame.cross(line_a, line_b, torus = False):
                                V_mat[a, b, s, t] = 0
                        else:
                            raise Exception("The undirected, torus case has not been implemented.")
        return V_mat


def display_classical(planar_game, print_strategy=False, torus = False):
    classical_value = planar_game.classical_value()
    planar_game.pred_mat = planar_game.value_matrix(directed=True, torus=torus)

    print(f"Classical value: {classical_value['classical_value']}")
    if print_strategy:
        print("Alice's classical strategy:")
        for idx, s in enumerate(planar_game.S):
            print(
                f"{s}: {np.array(planar_game.A[int(classical_value['alice_strategy'][idx])]['endpoints']).tolist()}, {planar_game.A[int(classical_value['alice_strategy'][idx])]['boundary_conditions']}"
            )
            # print("Bob's classical strategy:")
            # for idx_b, t in enumerate(planar_game.T):
            #     print(f"{t}: {np.array(planar_game.B[int(classical_value['bob_strategy'][idx_b])]['endpoints']).tolist()}, {planar_game.B[int(classical_value['bob_strategy'][idx_b])]['boundary_conditions']}")
            #     print(f"{planar_game.pred_mat[int(classical_value['alice_strategy'][idx]), int(classical_value['bob_strategy'][idx_b]), idx, idx_b]=}")
        print("Bob's classical strategy:")
        for idx, s in enumerate(planar_game.T):
            print(f"{s}: {np.array(planar_game.B[int(classical_value['bob_strategy'][idx])]['endpoints']).tolist()}, {planar_game.B[int(classical_value['bob_strategy'][idx])]['boundary_conditions']}")

        

"""
Print the quantum value of the planar game
Parameters:
* planar_game (PlanarGame) : Planar game to calculate quantum value for
* print_strategy (bool) : indicates whether to print out Alice's and Bob's quantum strategies, as POVMs 
"""


def display_quantum(planar_game, print_strategy=False, dim=2, iters=5):
    quantum_lower_bound = planar_game.quantum_value_lower_bound(dim=dim, iters=iters)
    print(f"Quantum value lower bound: {quantum_lower_bound['quantum_lower_bound']}")
    complex_formatter = {
            "complexfloat": lambda x: (
                f"{x.real} + {x.imag}j" if np.abs(x.imag) >= 10**(-6) else f"{x.real}"
            )
        }

    if print_strategy:
        for s_idx, s in enumerate(planar_game.S):
            print(f"{s}:")
            print("Alice's POVMs:")
            a_povm_sum = 0
            for a_idx, a in enumerate(planar_game.A):
                a_povm = np.array(
                    quantum_lower_bound["alice_strategy"][s_idx, a_idx].value
                ).round(3)
                a_povm_sum += a_povm
                if np.any(a_povm):
                    print(f"{np.array(a['endpoints']).tolist()}, {a['boundary_conditions']}:\n{np.array2string(
                        a_povm,
                        formatter=complex_formatter,
                    )}")
            print(f"Sum of Alice's POVMs: {np.array2string(a_povm_sum.round(3), formatter=complex_formatter)}")
            b_povm_sum = 0
            print("Bob's POVMs:")
            for b_idx, b in enumerate(planar_game.B):
                b_povm = np.array(
                    quantum_lower_bound["bob_strategy"][s_idx, b_idx].value
                ).round(3)
                b_povm_sum += b_povm
                if np.any(b_povm):
                    print(f"{np.array(b['endpoints']).tolist()}, {b['boundary_conditions']}:\n{np.array2string(
                        b_povm,
                        formatter=complex_formatter,
                    )}")
            print(f"Sum of Bob's POVMs: {np.array2string(b_povm_sum.round(3), formatter=complex_formatter)}")


def small_embedding_values():
    small_S = []
    small_S.append([(1,2), (2,3), (1,4), (1,5)])#, (2,4), (3,4), (5,4), (6,4)])  # , (3, 1), (3, 4)])
    quantum = False
    classical = True
    ns = False
    torus = False
    for S in small_S:
        for m, n in [(1, 3)]:  # , (1, 3), (1, 4), (2, 2)]:
            print(f"{S=}, {m=}, {n=}")
            planar_game = PlanarGame(S=S, n=n, m=m, torus = torus)
            if ns:
                print(f"{planar_game.nonsignaling_value()=}")
            if quantum:
                dim = 2
                iters = 3
                display_quantum(planar_game, print_strategy=True, dim=dim, iters=iters)
            if classical:
                display_classical(planar_game, print_strategy=True, torus = torus)

def cluster():
    planar_game = PlanarGame(S=eval(args.edges), n=args.n, m=args.m)
    print(f"{planar_game.S=}, {planar_game.m=}, {planar_game.n=}")
    print(f"{planar_game.nonsignaling_value()=}")


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
