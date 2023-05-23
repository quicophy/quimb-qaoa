"""
Implementation of the problem Hamiltonian of QAOA for different problems.
"""


import quimb as qu

from .gates import *


def hamiltonian_ops(G, problem="nae3sat"):
    """
    Returns a list of the operators composing the problem Hamiltonian for QAOA in order to compute the local expectation based on user input.
    """

    if problem == "nae3sat":
        return Nae3sat_Hamiltonian(G).operators()
    
    elif problem == "genome":
        return Genome_Hamiltonian(G).operators()
    
    else:
        raise ValueError("This problem is not implemented yet.")
    
def hamiltonian_gates(G, problem="nae3sat"):
    """
    Returns a list of the gates composing the problem Hamiltonian for QAOA in order to create a circuit based on user input.
    """

    if problem == "nae3sat":
        return Nae3sat_Hamiltonian(G).gates()
    
    elif problem == "genome":
        return Genome_Hamiltonian(G).gates()
    
    else:
        raise ValueError("This problem is not implemented yet.")


class Nae3sat_Hamiltonian:
    """
    Implementation of the problem Hamiltonian for the NAE 3-SAT problem.
    """

    def __init__(self, G):
        
        self.G = G

    def operators(self):

        n = len(self.G.edges)
        A = 1

        ops = [A*qu.pauli('Z') & qu.pauli('Z') for i in range(n)]

        qubits = self.G.edges

        return ops, qubits
    
    def gates(self):

        n = len(self.G.edges)
        A = 1

        coefs = [A for i in range(n)]
        
        ops = ["rzz" for i in range(n)]

        qubits = self.G.edges

        return coefs, ops, qubits
    

class Genome_Hamiltonian:
    """
    Implementation of the problem Hamiltonian for the genome assembly/travelling salesman problem.
    """

    def __init__(self, G):
        
        self.G = G

    def operators(self):

        n = self.G.numnodes
        w = 1
        d = 1

        ops = []
        qubits = []

        # Hamiltonian 1
        for i in range(n**2):
            ops.append(w*qu.pauli('Z'))
            qubits.append(i)

        # Hamiltonian 2
        for r in range(n):
            for i in range(n):
                for j in range(i):
                    ops.append(w/2*qu.pauli('Z') & qu.pauli('Z'))
                    qubits.append((i*n+r), (j*n+r))

                    ops.append(-w/2*qu.pauli('Z'))
                    qubits.append(i*n+r)

                    ops.append(-w/2*qu.pauli('Z'))
                    qubits.append(j*n+r)

        # Hamiltonian 3
        for i in range(n):
            for r in range(n):
                for s in range(r):
                    ops.append(w/2*qu.pauli('Z') & qu.pauli('Z'))
                    qubits.append((i*n+r), (i*n+s))

                    ops.append(-w/2*qu.pauli('Z'))
                    qubits.append(i*n+r)

                    ops.append(-w/2*qu.pauli('Z'))
                    qubits.append(i*n+s)

        # Hamiltonian 4
        for i in range(n):
            for j in range(n):
                for r in range(n):

                    if j != i:
                        continue

                    s = (r+1)%n

                    ops.append(d/4*qu.pauli('Z') & qu.pauli('Z'))
                    qubits.append((i*n+r), (j*n+s))

                    ops.append(-d/4*qu.pauli('Z'))
                    qubits.append(i*n+r)

                    ops.append(-w/2*qu.pauli('Z'))
                    qubits.append(j*n+s)

        return ops, qubits
    
    def gates(self):

        n = self.G.numnodes
        w = 1
        d = 1

        coefs = []
        ops = []
        qubits = []

        # Hamiltonian 1
        for i in range(n**2):
            coefs.append(w)
            ops.append("rz")
            qubits.append(i)

        # Hamiltonian 2
        for r in range(n):
            for i in range(n):
                for j in range(i):
                    coefs.append(w/2)
                    ops.append("rzz")
                    qubits.append((i*n+r), (j*n+r))

                    coefs.append(-w/2)
                    ops.append("rz")
                    qubits.append(i*n+r)

                    coefs.append(-w/2)
                    ops.append("rz")
                    qubits.append(j*n+r)

        # Hamiltonian 3
        for i in range(n):
            for r in range(n):
                for s in range(r):
                    coefs.append(w/2)
                    ops.append("rzz")
                    qubits.append((i*n+r), (i*n+s))

                    coefs.append(-w/2)
                    ops.append("rz")
                    qubits.append(i*n+r)

                    coefs.append(-w/2)
                    ops.append("rz")
                    qubits.append(i*n+s)

        # Hamiltonian 4
        for i in range(n):
            for j in range(n):
                for r in range(n):

                    if j != i:
                        continue

                    s = (r+1)%n

                    coefs.append(d/4)
                    ops.append("rzz")
                    qubits.append((i*n+r), (j*n+s))

                    coefs.append(-d/4)
                    ops.append("rz")
                    qubits.append(i*n+r)

                    coefs.append(-d/4)
                    ops.append("rz")
                    qubits.append(j*n+s)

        return coefs, ops, qubits