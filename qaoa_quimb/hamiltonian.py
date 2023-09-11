"""
Implementation of the problem Hamiltonian of QAOA for different problems.
"""


import quimb as qu


def hamiltonian(G, problem):
    """
    Returns a list of the operators composing the problem Hamiltonian for QAOA in order to compute the local expectation based on user input.
    """

    if problem == "nae3sat":
        return IsingHamiltonian(G)

    elif problem == "mono1in3sat":
        return IsingWithFieldHamiltonian(G)

    elif problem == "mono2sat":
        return IsingWithFieldHamiltonian(G)

    elif problem == "maxcut":
        return IsingHamiltonian(G)

    elif problem == "genome":
        return GenomeHamiltonian(G)

    else:
        raise ValueError("This problem is not implemented yet.")


class IsingHamiltonian:
    """
    Implementation of the problem Hamiltonian for the NAE 3-SAT problem.
    """

    def __init__(self, G):
        self.G = G

        rzz_gates = self.cost_hamiltonian()

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

        return ops, qubits

    def gates(self):
        qubits = []
        ops = []
        coefs = []

        for qubit, value in self.rzz_gates.items():
            qubits.append(qubit)
            ops.append("rzz")
            coefs.append(-1 * value)

        return coefs, ops, qubits


class IsingWithFieldHamiltonian:
    """
    Implementation of the problem Hamiltonian for the NAE 3-SAT problem.
    """

    def __init__(self, G):
        self.G = G

        rz_gates, rzz_gates = self.cost_hamiltonian()

        self.rz_gates = rz_gates
        self.rzz_gates = rzz_gates

    @property
    def numqubit(self):
        n = self.G.numnodes
        return n

    def cost_hamiltonian(self):
        rz_gates = {}
        rzz_gates = {}

        for edge, weight in list(self.G.terms.items()):
            if len(edge) == 2:
                rzz_gates[edge] = weight
            elif len(edge) == 1:
                rz_gates[edge] = weight

        return rz_gates, rzz_gates

    def operators(self):
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
        qubits = []
        ops = []
        coefs = []
        localham_rzz = {}
        localham_rz = {}

        for qubit, value in self.rzz_gates.items():
            qubits.append(qubit)
            ops.append("rzz")
            coefs.append(-1 * value)
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
            coefs.append(2 * value)
            # localham_rz[qubit[0]] = rz_gate_param_gen([value*gamma])

        for qubit, value in self.rzz_gates.items():
            qubits.append(qubit)
            ops.append("rzz")
            coefs.append(-1 * value)
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
