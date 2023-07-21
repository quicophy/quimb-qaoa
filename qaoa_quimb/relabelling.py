"""
Functions for relabelling qubits for more efficient contractions.
"""

import numpy as np
from matrex import msro


def relabelling(numqubit, gates):
    """
    ONLY FOR RZZ GATES. DOESN'T WORK.
    """
    
    labels = np.zeros((len(gates), numqubit))

    for iter, qubits in enumerate(list(gates.keys())):

        labels[iter, qubits[0]] = 1
        labels[iter, qubits[1]] = 1

    # reordered_idx = msro(labels)
    reordered_idx = msro(labels.T)

    new_gates = {}

    for qubits, weight in gates.items():
        
        new_qubits = (reordered_idx[qubits[0]], reordered_idx[qubits[1]])
        new_gates[new_qubits] = weight

    return new_gates
        


# import networkx as nx
# from qaoa_quimb.hamiltonian import Nae3satHamiltonian

# G = nx.random_regular_graph(3, 6, seed=123)
# G.numnodes = G.order()
# G.terms = {(i, j): 1 for i, j in G.edges}

# hamiltonian = Nae3satHamiltonian(G)
# numqubit = hamiltonian.numqubit
# gates = hamiltonian.rzz_gates

# relabelling(numqubit, gates)