"""
Implementation of different types of circuits for QAOA.
"""


import quimb as qu
import quimb.tensor as qtn

from .hamiltonian import hamiltonian_gates


def create_qaoa_circ(G, p, gammas, betas, qaoa_version, problem="nae3sat"):
    """
    Creates the correct qaoa circuit based on user input.
    """

    if qaoa_version == 'regular':
        qc = create_regular_qaoa_circ(G, p, gammas, betas, problem=problem)
    elif qaoa_version == 'gm':
        qc = create_gm_qaoa_circ(G, p, gammas, betas, problem=problem)
    else:
        raise ValueError('The QAOA version is not valid.')

    return qc

def create_regular_qaoa_circ(G, p, gammas, betas, problem="nae3sat", **circuit_opts,):
    """
    Creates a parametrized regular qaoa circuit.

    Returns:
        circ: quantum circuit
    """

    circuit_opts.setdefault('gate_opts', {})
    circuit_opts['gate_opts'].setdefault('contract', False)

    n = G.numnodes

    gates = []

    # layer of hadamards to get into plus state
    for i in range(n):
        gates.append((0, 'h', i))

    coefs, ops, qubits = hamiltonian_gates(G, problem=problem)
    
    for d in range(p):

        # problem Hamiltonian
        for (coef, op, qubit) in zip(coefs, ops, qubits):
            gates.append((d, op, coef * gammas[d], *qubit))

        # mixer Hamiltonian
        for i in range(n):
            gates.append((d, 'rx', -betas[d] * 2, i))

    circ = qtn.Circuit(n, **circuit_opts)
    circ.apply_gates(gates)

    # print(circ.psi)

    # circ.psi.compress_all()
    # print(circ.psi)

    return circ

def create_gm_qaoa_circ(G, p, gammas, betas, problem="nae3sat", **circuit_opts,):
    """
    ONLY VALID UP TO 14 QUBITS. Creates a parametrized grover-mixer qaoa circuit.
    """

    circuit_opts.setdefault('gate_opts', {})
    circuit_opts['gate_opts'].setdefault('contract', False)

    n = G.numnodes

    circ = qtn.Circuit(n, **circuit_opts)

    gates = []

    # layer of hadamards to get into plus state
    for i in range(n):
        gates.append((0, 'h', i))

    circ.apply_gates(gates)

    for d in range(p):

        gates = []

        # problem Hamiltonian
        coefs, ops, qubits = hamiltonian_gates(G, problem=problem)

        for (coef, op, qubit) in zip(coefs, ops, qubits):
            gates.append((d, op, coef * gammas[d], *qubit))

        # mixer Hamiltonian
        for i in range(n):
            gates.append((d, 'h', i))
            gates.append((d, 'x', i))

        circ.apply_gates(gates)

        C = qu.ncontrolled_gate(n-1, qu.rotation(-betas[d]*2), sparse=False)

        circ.apply_gate_raw(C, range(0,n), gate_round=d)

        gates = []

        for i in range(n):
            gates.append((d, 'x', i))
            gates.append((d, 'h', i))

        circ.apply_gates(gates)

    return circ