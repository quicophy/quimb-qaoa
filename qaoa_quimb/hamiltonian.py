"""
Implementation of the problem Hamiltonian of QAOA for different problems.
"""


import quimb as qu

from .relabelling import relabelling
from .gates import *


def hamiltonian(G, problem):
    """
    Returns a list of the operators composing the problem Hamiltonian for QAOA in order to compute the local expectation based on user input.
    """

    if problem == "nae3sat":
        return Nae3satHamiltonian(G)
    
    elif problem == "2sat":
        return Mono2satHamiltonian(G)
    
    elif problem == "maxcut":
        return Nae3satHamiltonian(G)

    elif problem == "genome":
        return GenomeHamiltonian(G)

    else:
        raise ValueError("This problem is not implemented yet.")


class Mono2satHamiltonian:
    """
    Implementation of the problem Hamiltonian for the NAE 3-SAT problem.
    """

    def __init__(self, G):
        self.G = G

        rzz_gates = self.cost_hamiltonian()
        rzz_gates = relabelling(self.numqubit, rzz_gates)
        
        self.rzz_gates = rzz_gates

    @property
    def numqubit(self):
        n = self.G.numnodes
        return n

    def cost_hamiltonian(self):
        rzz_gates = {}

        for edge, weight in list(self.G.terms.items()):
            rzz_gates[edge] = weight

        return rzz_gates

    def operators(self):
        qubits = []
        ops = []

        for qubit, value in self.rzz_gates.items():
            qubits.append(qubit)
            ops.append(value * qu.pauli("Z") & qu.pauli("Z"))

        # print(qubits)

        return ops, qubits

    def gates(self, gamma):
        qubits = []
        ops = []
        coefs = []

        for qubit, value in self.rzz_gates.items():
            qubits.append(qubit)
            # ops.append("rzz")
            ops.append(rzz_param_gen([-value*gamma]))
            coefs.append(-value)

        return coefs, ops, qubits


class Nae3satHamiltonian:
    """
    Implementation of the problem Hamiltonian for the NAE 3-SAT problem.
    """

    def __init__(self, G):
        self.G = G

        rzz_gates = self.cost_hamiltonian()
        rzz_gates = relabelling(self.numqubit, rzz_gates)
        
        self.rzz_gates = rzz_gates

    @property
    def numqubit(self):
        n = self.G.numnodes
        return n

    def cost_hamiltonian(self):
        rzz_gates = {}

        for edge, weight in list(self.G.terms.items()):
            rzz_gates[edge] = weight

        return rzz_gates

    def operators(self):
        qubits = []
        ops = []

        for qubit, value in self.rzz_gates.items():
            qubits.append(qubit)
            ops.append(value * qu.pauli("Z") & qu.pauli("Z"))

        # print(qubits)

        return ops, qubits

    def gates(self):
        qubits = []
        ops = []
        coefs = []

        for qubit, value in self.rzz_gates.items():
            qubits.append(qubit)
            ops.append("rzz")
            coefs.append(-value)

        return coefs, ops, qubits


class GenomeHamiltonian:
    """
    Implementation of the problem Hamiltonian for the genome assembly/travelling salesman problem.
    """

    def __init__(self, G):
        self.G = G
        rz_gates, rzz_gates = self.cost_hamiltonian()
        self.rz_gates = rz_gates
        self.rzz_gates = rzz_gates

    @property
    def numqubit(self):
        n = self.G.numnodes**2
        return n

    def cost_hamiltonian(self):
        n = self.G.numnodes

        rz_gates = {}
        rzz_gates = {}

        # Hamiltonian 1
        for i in range(n**2):
            rz_gates[(i,)] = rz_gates.get((i,), 0) + 1 / 2

        # Hamiltonian 2
        for j in range(1, n):
            for v in range(n):
                for s in range(j):
                    if j == s:
                        continue

                    rz_gates[(v * n + j,)] = rz_gates.get((v * n + j,), 0) - 1 / 4

                    rz_gates[(v * n + s,)] = rz_gates.get((v * n + s,), 0) - 1 / 4

                    rzz_gates[(v * n + j, v * n + s)] = (
                        rzz_gates.get((v * n + j, v * n + s), 0) + 1 / 4
                    )

        # Hamiltonian 3
        for j in range(n):
            for v in range(1, n):
                for s in range(v):
                    if v == s:
                        continue

                    rz_gates[(v * n + j,)] = rz_gates.get((v * n + j,), 0) - 1 / 4

                    rz_gates[(s * n + j,)] = rz_gates.get((s * n + j,), 0) - 1 / 4

                    rzz_gates[(v * n + j, s * n + j)] = (
                        rzz_gates.get((v * n + j, s * n + j), 0) + 1 / 4
                    )

        keys = self.G.terms.keys()
        # Hamiltonian 4
        for u in range(n):
            for v in range(n):
                for j in range(n - 1):
                    if (u, v) in keys or (v, u) in keys:
                        continue

                    if u == v:
                        continue

                    if u * n + j == v * n + s:
                        continue

                    s = (j + 1) % n

                    rz_gates[(u * n + j,)] = rz_gates.get((u * n + j,), 0) - 1 / 4

                    rz_gates[(v * n + s,)] = rz_gates.get((v * n + s,), 0) - 1 / 4

                    rzz_gates[(u * n + j, v * n + s)] = (
                        rzz_gates.get((u * n + j, v * n + s), 0) + 1 / 4
                    )

        return rz_gates, rzz_gates

    def operators(self):
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
        qubits = []
        ops = []
        coefs = []
        # localham_rzz = {}
        # localham_rz = {}

        for qubit, value in self.rz_gates.items():
            qubits.append(qubit)
            ops.append("rz")
            coefs.append(value)
            # localham_rz[qubit[0]] = rz_gate_param_gen([value*gamma])

        for qubit, value in self.rzz_gates.items():
            qubits.append(qubit)
            ops.append("rzz")
            coefs.append(value)
            # localham_rzz[qubit] = rzz_param_gen([value*gamma]).reshape((4,4))
        
        # print(np.shape(rz_gate_param_gen([value*gamma])))
        # print(np.shape(rzz_param_gen([value*gamma])))
        # print(localham_rz.keys())
        # print(localham_rzz.keys())

        # localham = qu.tensor.LocalHamGen(localham_rzz, H1=localham_rz)
        # # localham.draw()

        # qubits = []
        # ops = []
        # coefs = []

        # for qubit, op in localham.items():
        #     qubits.append(qubit)
        #     ops.append(op)
        #     coefs.append(None)

        return coefs, ops, qubits
