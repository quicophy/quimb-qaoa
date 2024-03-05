"""
Implementation of the problem Hamiltonian of QAOA for different problems.
"""


import numpy as np
import networkx as nx
import quimb as qu


def hamiltonian(graph):
    """
    Create the problem Hamiltonian for a given problem.

    Parameters:
    -----------
    graph: ProblemGraph
        Graph representing the instance of the problem

    Returns:
    --------
    hamiltonian: Hamiltonian
        Hamiltonian of the problem instance.

    """

    if graph.problem == "maxcut" or graph.problem == "nae3sat":
        return IsingHamiltonian(graph)

    elif graph.problem == "2sat" or graph.problem == "1in3sat":
        return IsingWithFieldHamiltonian(graph)

    # elif graph.problem == "genome":
    #     return _GenomeHamiltonian(graph)

    else:
        raise ValueError("The problem given is not implemented yet.")


class IsingHamiltonian:
    r"""
    Implementation of the Ising Hamiltonian without local field

    .. math::

        H = \sum_{i, j} J_{ij} \sigma_i^z \sigma_j^z,

    where interaction terms are given by the problem instance. The method operator() returns the necessary list of operators for the contraction of the QAOA circuit (see contraction.py). The method gates() returns the necessary list of gates for the QAOA circuit and QAOA MPS creation (see circuit.py and mps.py).

    Parameters:
    -----------
    G: ProblemGraph
        Graph representing the instance of the problem.
    """

    def __init__(self, graph):
        self.graph = graph

        # generate the necessary RZZ gates
        rzz_gates = self.problem_hamiltonian()

        self.rzz_gates = rzz_gates

    @property
    def numqubit(self):
        """Number of qubits to represent the problem."""
        return self.graph.numnodes

    def problem_hamiltonian(self):
        """
        Computes a dictionnary of RZZ gates with its parameter representing the problem Hamiltonian.

        Returns:
        --------
        rzz_gates: dict[tuple[int, int], float]
            RZZ gates representing the problem Hamiltonian, where the keys are the edges of the graph and the values are the parameters of the RZZ gates.
        """

        rzz_gates = {}

        for edge, weight in list(self.graph.terms.items()):
            rzz_gates[edge] = weight

        return rzz_gates

    def operators(self):
        """
        Returns a list of the operators the problem Hamiltonian. Necessary for the contraction of the QAOA circuit (see contraction.py).

        Returns:
        --------
        ops: list[str]
            List of operators of problem Hamiltonian.
        qubits: list[Tuple[int]]
            List of qubits on which the gates act.
        """

        qubits = []
        ops = []

        for qubit, value in self.rzz_gates.items():
            qubits.append(qubit)
            ops.append(value * qu.pauli("Z") & qu.pauli("Z"))

        return ops, qubits

    def gates(self):
        """
        Returns the gates of the problem Hamiltonian. Necessary for the QAOA circuit and QAOA MPS creation (see circuit.py and mps.py).

        Returns:
        --------
        coefs: list[float]
            List of coefficients of the gates.
        ops: list[str]
            List of gates.
        qubits: list[Tuple[int]]
            List of qubits on which the gates act.
        """

        qubits = []
        ops = []
        coefs = []

        for qubit, value in self.rzz_gates.items():
            qubits.append(qubit)
            ops.append("rzz")
            coefs.append(1 / 2 * value)

        return coefs, ops, qubits


class IsingWithFieldHamiltonian:
    """
    Implementation of the Ising Hamiltonian with local field

    .. math::

        H = \sum_{i, j} J_{ij} \sigma_i^z \sigma_j^z + \sum_i h_i \sigma_i^z,

    where interactions and field terms are given by the problem instance. The method operator() returns the necessary list of operators for the contraction of the QAOA circuit (see contraction.py). The method gates() returns the necessary list of gates for the QAOA circuit and QAOA MPS creation (see circuit.py and mps.py).

    Parameters:
    -----------
    G: ProblemGraph
        Graph representing the instance of the problem.
    """

    def __init__(self, graph):
        self.graph = graph

        # generate the necessary RZ and RZZ gates
        rz_gates, rzz_gates = self.problem_hamiltonian()

        self.rz_gates = rz_gates
        self.rzz_gates = rzz_gates

    @property
    def numqubit(self):
        """Number of qubits to represent the problem."""
        return self.graph.numnodes

    def problem_hamiltonian(self):
        """
        Computes a dictionnary of RZ and RZZ gates with its parameter representing the problem Hamiltonian.

        Returns:
        --------
        rz_gates: dict[int, float]
            RZ gates representing the interaction terms of problem Hamiltonian, where the keys are the qubits and the values are the parameters of the RZ gates.
        rzz_gates: dict[tuple[int, int], float]
            RZZ gates representing the field terms of the problem Hamiltonian, where the keys are the edges of the graph and the values are the parameters of the RZZ gates.
        """

        rz_gates = {}
        rzz_gates = {}

        for edge, weight in list(self.graph.terms.items()):
            if len(edge) == 2:
                rzz_gates[edge] = weight
            elif len(edge) == 1:
                rz_gates[edge] = weight

        return rz_gates, rzz_gates

    def operators(self):
        """
        Returns a list of the operators the problem Hamiltonian. Necessary for the contraction of the QAOA circuit (see contraction.py).

        Returns:
        --------
        ops: list[str]
            List of operators of problem Hamiltonian.
        qubits: list[Tuple[int]]
            List of qubits on which the gates act.
        """

        qubits = []
        ops = []
        localham_rz = {}
        localham_rzz = {}

        for qubit, value in self.rzz_gates.items():
            qubits.append(qubit)
            ops.append(value * qu.pauli("Z") & qu.pauli("Z"))
            localham_rzz[qubit] = value * qu.pauli("Z") & qu.pauli("Z")

        for qubit, value in self.rz_gates.items():
            qubits.append(qubit)
            ops.append(value * qu.pauli("Z"))
            localham_rz[qubit[0]] = value * qu.pauli("Z")

        # for qubit, value in self.rzz_gates.items():
        #     qubits.append(qubit)
        #     ops.append((qu.eye(4) + (qu.pauli("Z") & qu.eye(2))) @ (qu.eye(4) + (qu.eye(2) & qu.pauli("Z"))))

        localham = qu.tensor.LocalHamGen(localham_rzz, H1=localham_rz)

        qubits = []
        ops = []

        for qubit, op in localham.items():
            qubits.append(qubit)
            ops.append(op)

        return ops, qubits

    def gates(self):
        """
        Returns the gates of the problem Hamiltonian. Necessary for the QAOA circuit and QAOA MPS creation (see circuit.py and mps.py).

        Returns:
        --------
        coefs: list[float]
            List of coefficients of the gates.
        ops: list[str]
            List of gates.
        qubits: list[Tuple[int]]
            List of qubits on which the gates act.
        """

        qubits = []
        ops = []
        coefs = []
        localham_rzz = {}
        localham_rz = {}

        for qubit, value in self.rzz_gates.items():
            qubits.append(qubit)
            ops.append("rzz")
            coefs.append(2 * value)
        #     localham_rzz[qubit] = rzz_param_gen([value*gamma]).reshape((4,4))

        for qubit, value in self.rz_gates.items():
            qubits.append(qubit)
            ops.append("rz")
            coefs.append(2 * value)
            # localham_rz[qubit[0]] = rz_gate_param_gen([value*gamma])

        # localham = qu.tensor.LocalHamGen(localham_rzz, H1=localham_rz)

        # qubits = []
        # ops = []
        # coefs = []

        # import numpy as np
        # for qubit, op in localham.items():
        #     qubits.append(qubit)
        #     ops.append(op)
        #     coefs.append(None)

        return coefs, ops, qubits


class _GenomeHamiltonian:
    """
    UNTESTED. Implementation of the problem Hamiltonian for the Genome Assembly/Travelling Salesman Problem (TSP).

    Parameters:
    -----------
    G: ProblemGraph
        Graph representing the instance of the problem.
    """

    def __init__(self, graph):
        self.graph = graph

        # generate the necessary RZ and RZZ gates
        rz_gates, rzz_gates = self.problem_hamiltonian()

        self.rz_gates = rz_gates
        self.rzz_gates = rzz_gates

    @property
    def numqubit(self):
        """Number of qubits to represent the problem."""
        return (self.graph.numnodes - 1) ** 2

    def problem_hamiltonian(self):
        """
        Computes a dictionnary of RZ and RZZ gates with its parameter representing the problem Hamiltonian.

        Returns:
        --------
        rz_gates: dict[int, float]
            RZ gates representing the interaction terms of problem Hamiltonian, where the keys are the qubits and the values are the parameters of the RZ gates.
        rzz_gates: dict[tuple[int, int], float]
            RZZ gates representing the field terms of the problem Hamiltonian, where the keys are the edges of the graph and the values are the parameters of the RZZ gates.
        """

        n = self.graph.numnodes - 1

        adj = 100 * nx.to_numpy_array(self.G)
        deg = np.sum(adj, axis=1).tolist()
        B = 0.0009
        rz_gates = {}
        rzz_gates = {}

        # Hamiltonian 1
        for i in range(n**2):
            rz_gates[(i,)] = rz_gates.get((i,), 0) + 1

        # Hamiltonian 2
        for j in range(1, n):
            for v in range(n):
                for s in range(j):
                    if j == s:
                        continue
                    rz_gates[(v * n + j,)] = rz_gates.get((v * n + j,), 0) - 1 / 2

                    rz_gates[(v * n + s,)] = rz_gates.get((v * n + s,), 0) - 1 / 2

                    rzz_gates[(v * n + j, v * n + s)] = (
                        rzz_gates.get((v * n + j, v * n + s), 0) + 1 / 2
                    )

        # Hamiltonian 3
        for j in range(n):
            for v in range(1, n):
                for s in range(v):
                    if v == s:
                        continue

                    rz_gates[(v * n + j,)] = rz_gates.get((v * n + j,), 0) - 1 / 2

                    rz_gates[(s * n + j,)] = rz_gates.get((s * n + j,), 0) - 1 / 2

                    rzz_gates[(v * n + j, s * n + j)] = (
                        rzz_gates.get((v * n + j, s * n + j), 0) + 1 / 2
                    )

        # Hamiltonian 4
        for u in range(n):
            for v in range(n):
                # for (u, v), wuv in terms.items():
                wuv = adj[u, v]
                for j in range(n - 1):
                    # if (u, v) in keys or (v, u) in keys:
                    # continue

                    if u == v:
                        continue

                    if u * n + j == v * n + s:
                        continue

                    s = (j + 1) % n

                    rz_gates[(u * n + j,)] = rz_gates.get((u * n + j,), 0) + B * wuv / 4

                    rz_gates[(v * n + s,)] = rz_gates.get((v * n + s,), 0) + B * wuv / 4

                    rzz_gates[(u * n + j, v * n + s)] = (
                        rzz_gates.get((u * n + j, v * n + s), 0) - B * wuv / 4
                    )
        for u in range(1, n):
            #     # for (u, v), wuv in terms.items():
            wuv = adj[0, u]
            for j in [1, n - 1]:
                # if (u, v) in keys or (v, u) in keys:
                #         # continue

                #         s = (j + 1) % n
                rz_gates[(u * j,)] = rz_gates.get((u * j,), 0) + B * wuv / 4

        return rz_gates, rzz_gates

    def operators(self):
        """
        Returns a list of the operators the problem Hamiltonian. Necessary for the contraction of the QAOA circuit (see contraction.py).

        Returns:
        --------
        ops: list[str]
            List of operators of problem Hamiltonian.
        qubits: list[Tuple[int]]
            List of qubits on which the gates act.
        """

        qubits = []
        ops = []
        localham_rz = {}
        localham_rzz = {}

        for qubit, value in self.rz_gates.items():
            qubits.append(qubit)
            ops.append(value * qu.pauli("Z"))
            localham_rz[qubit[0]] = value * qu.pauli("Z")

        for qubit, value in self.rzz_gates.items():
            qubits.append(qubit)
            ops.append(value * qu.pauli("Z") & qu.pauli("Z"))
            localham_rzz[qubit] = value * qu.pauli("Z") & qu.pauli("Z")

        localham = qu.tensor.LocalHamGen(localham_rzz, H1=localham_rz)

        qubits = []
        ops = []

        for qubit, op in localham.items():
            qubits.append(qubit)
            ops.append(op)

        return ops, qubits

    def gates(self):
        """
        Returns the gates of the problem Hamiltonian. Necessary for the QAOA circuit and QAOA MPS creation (see circuit.py and mps.py).

        Returns:
        --------
        coefs: list[float]
            List of coefficients of the gates.
        ops: list[str]
            List of gates.
        qubits: list[Tuple[int]]
            List of qubits on which the gates act.
        """

        qubits = []
        ops = []
        coefs = []
        # localham_rzz = {}
        # localham_rz = {}

        for qubit, value in self.rz_gates.items():
            qubits.append(qubit)
            ops.append("rz")
            coefs.append(2 * value)
            # localham_rz[qubit[0]] = rz_gate_param_gen([value*gamma])

        for qubit, value in self.rzz_gates.items():
            qubits.append(qubit)
            ops.append("rzz")
            coefs.append(-value)
            # localham_rzz[qubit] = rzz_param_gen([value*gamma]).reshape((4,4))

        # localham = qu.tensor.LocalHamGen(localham_rzz, H1=localham_rz)

        # qubits = []
        # ops = []
        # coefs = []

        # for qubit, op in localham.items():
        #     qubits.append(qubit)
        #     ops.append(op)
        #     coefs.append(None)

        return coefs, ops, qubits
