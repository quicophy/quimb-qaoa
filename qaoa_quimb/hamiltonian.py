"""
Implementation of the problem Hamiltonian of QAOA for different problems.
"""


import quimb as qu

from .gates import *


def hamiltonian(G, problem):
    """
    Returns a list of the operators composing the problem Hamiltonian for QAOA in order to compute the local expectation based on user input.
    """

    if problem == "nae3sat":
        return Nae3sat_Hamiltonian(G)

    elif problem == "genome":
        return Genome_Hamiltonian(G)

    else:
        raise ValueError("This problem is not implemented yet.")


class Nae3sat_Hamiltonian:
    """
    Implementation of the problem Hamiltonian for the NAE 3-SAT problem.
    """

    def __init__(self, G):
        self.G = G

    @property
    def numqubit(self):
        n = self.G.numnodes
        return n

    def operators(self):
        ops = []
        qubits = []
        for edge, weight in list(self.G.terms.items()):
            ops.append(weight * qu.pauli("Z") & qu.pauli("Z"))
            qubits.append(edge)

        return ops, qubits

    def gates(self):
        coefs = []
        ops = []
        qubits = []
        for edge, weight in list(self.G.terms.items()):
            coefs.append(-weight)
            ops.append("rzz")
            qubits.append(edge)

        return coefs, ops, qubits


class Genome_Hamiltonian:
    """
    Implementation of the problem Hamiltonian for the genome assembly/travelling salesman problem.
    """

    def __init__(self, G):
        self.G = G
        rz_gates, rzz_gates = self.__cost_hamiltonian__()
        self.rz_gates = rz_gates
        self.rzz_gates = rzz_gates

    def __cost_hamiltonian__(self):
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

    @property
    def numqubit(self):
        n = self.G.numnodes**2
        return n

    def operators(self):
        qubits = []
        ops = []

        for qubit, value in self.rz_gates.items():
            qubits.append(qubit)
            ops.append(value * qu.pauli("Z"))

        for qubit, value in self.rzz_gates.items():
            qubits.append(qubit)
            ops.append(value * qu.pauli("Z") & qu.pauli("Z"))

        return ops, qubits

    def gates(self):
        qubits = []
        ops = []
        coefs = []

        for qubit, value in self.rz_gates.items():
            qubits.append(qubit)
            ops.append("rz")
            coefs.append(value)

        for qubit, value in self.rzz_gates.items():
            qubits.append(qubit)
            ops.append("rzz")
            coefs.append(value)

        return coefs, ops, qubits
