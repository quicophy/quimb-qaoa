"""
Implementation of different types of circuits for QAOA.
"""


import quimb as qu
import quimb.tensor as qtn
import numpy as np

from .hamiltonian import hamiltonian


def create_qaoa_circ(G, p, gammas, betas, qaoa_version, problem="nae3sat"):
    """
    Creates the correct qaoa circuit based on user input.
    """

    if qaoa_version == "regular":
        qc = create_regular_qaoa_circ(G, p, gammas, betas, problem=problem)
    elif qaoa_version == "gm":
        qc = create_gm_qaoa_circ(G, p, gammas, betas, problem=problem)
    elif qaoa_version == "qgm":
        qc = create_qgm_qaoa_circ(G, p, gammas, betas, problem=problem)
    elif qaoa_version == "xy":
        qc = create_xy_mixer_qaoa_circ(G, p, gammas, betas, problem=problem)
    else:
        raise ValueError("The QAOA version is not valid.")

    return qc


def create_regular_qaoa_circ(
    G,
    p,
    gammas,
    betas,
    problem="nae3sat",
    **circuit_opts,
):
    """
    Creates a parametrized regular qaoa circuit.

    Returns:
        circ: quantum circuit
    """

    circuit_opts.setdefault("gate_opts", {})
    circuit_opts["gate_opts"].setdefault("contract", False)

    hamil = hamiltonian(G, problem)
    n = hamil.numqubit

    circ = qtn.Circuit(n, **circuit_opts)

    gates = []

    # layer of hadamards to get into plus state
    for i in range(n):
        gates.append((0, "h", i))

    circ.apply_gates(gates)

    for d in range(p):
        gates = []

        # problem Hamiltonian
        coefs, ops, qubits = hamil.gates()

        for coef, op, qubit in zip(coefs, ops, qubits):
            # circ.apply_gate_raw(op, qubit, gate_round=d)
            gates.append((d, op, coef * gammas[d], *qubit))

        # mixer Hamiltonian
        for i in range(n):
            gates.append((d, "rx", 2 * betas[d], i))

        circ.apply_gates(gates)

    return circ


def create_gm_qaoa_circ(
    G,
    p,
    gammas,
    betas,
    problem="nae3sat",
    **circuit_opts,
):
    """
    ONLY WORKS UP TO AROUND 14 QUBITS. Creates a parametrized grover-mixer qaoa circuit.

    Returns:
        circ: quantum circuit
    """

    circuit_opts.setdefault("gate_opts", {})
    circuit_opts["gate_opts"].setdefault("contract", False)

    hamil = hamiltonian(G, problem)
    n = hamil.numqubit

    circ = qtn.Circuit(n, **circuit_opts)

    gates = []

    # layer of hadamards to get into plus state
    for i in range(n):
        gates.append((0, "h", i))

    circ.apply_gates(gates)

    for d in range(p):
        gates = []

        # problem Hamiltonian
        coefs, ops, qubits = hamil.gates()
        for coef, op, qubit in zip(coefs, ops, qubits):
            # circ.apply_gate_raw(op, qubit, gate_round=d)
            gates.append((d, op, coef * gammas[d], *qubit))

        # mixer Hamiltonian
        for i in range(n):
            gates.append((d, "h", i))
            gates.append((d, "x", i))

        circ.apply_gates(gates)

        ncrz_gate = np.eye(2**n, dtype=complex)
        ncrz_gate[-1, -1] = np.exp(-2j * betas[d])
        # ncrz_gate = qu.ncontrolled_gate(n - 1, qu.rotation(-betas[d] * 2), sparse=False)
        circ.apply_gate_raw(ncrz_gate, range(0, n), gate_round=d, tags="NCRZ")

        gates = []

        for i in range(n):
            gates.append((d, "x", i))
            gates.append((d, "h", i))

        circ.apply_gates(gates)

    return circ


def create_qgm_qaoa_circ(
    G,
    p,
    gammas,
    betas,
    problem="nae3sat",
    **circuit_opts,
):
    """
    Creates a parametrized quasi-grover-mixer qaoa circuit.

    Returns:
        circ: quantum circuit
    """

    circuit_opts.setdefault("gate_opts", {})
    circuit_opts["gate_opts"].setdefault("contract", False)

    hamil = hamiltonian(G, problem)
    n = hamil.numqubit

    circ = qtn.Circuit(n, **circuit_opts)

    gates = []

    # layer of hadamards to get into plus state
    for i in range(n):
        gates.append((0, "h", i))

    circ.apply_gates(gates)

    for d in range(p):
        gates = []

        # problem Hamiltonian
        coefs, ops, qubits = hamil.gates()

        for coef, op, qubit in zip(coefs, ops, qubits):
            # circ.apply_gate_raw(op, qubit, gate_round=d)
            gates.append((d, op, coef * gammas[d], *qubit))

        # mixer Hamiltonian
        for i in range(n):
            gates.append((d, "h", i))

        for coef, op, qubit in zip(coefs, ops, qubits):
            # circ.apply_gate_raw(op, qubit, gate_round=d)
            gates.append((d, op, -coef * betas[d], *qubit))

        for i in range(n):
            gates.append((d, "h", i))

        circ.apply_gates(gates)

    return circ


def create_xy_mixer_qaoa_circ(
    G,
    p,
    gammas,
    betas,
    problem="nae3sat",
    **circuit_opts,
):
    """
    Creates a parametrized regular qaoa circuit.

    Returns:
        circ: quantum circuit
    """

    circuit_opts.setdefault("gate_opts", {})
    circuit_opts["gate_opts"].setdefault("contract", False)

    hamil = hamiltonian(G, problem)
    n = hamil.numqubit

    dims = n * [2]

    wstate = qu.gen.states.w_state(n)

    mps = qtn.tensor_1d.MatrixProductState.from_dense(wstate, dims)

    circ = qtn.Circuit(n, psi0=mps, **circuit_opts)

    for d in range(p):
        gates = []
        # problem Hamiltonian
        coefs, ops, qubits = hamil.gates()

        for coef, op, qubit in zip(coefs, ops, qubits):
            # circ.apply_gate_raw(op, qubit, gate_round=d)
            gates.append((d, op, coef * gammas[d], *qubit))

        circ.apply_gates(gates)
        # mixer Hamiltonian

        for b in circ.sample(18):
            print(b)
        sn = int(n**1 / 2)
        for j in range(sn):
            rxy = np.array(
                [
                    [1, 0, 0, 0],
                    [0, -1j * np.sin(-betas[d]), np.cos(betas[d]), 0],
                    [0, np.cos(betas[d]), -1j * np.sin(-betas[d]), 0],
                    [0, 0, 0, 1],
                ]
            )
            v = (j + 1) % n
            circ.apply_gate_raw(rxy, (j, v), tags="iswap")

    return circ


def create_old_qgm_qaoa_circ(
    G,
    p,
    gammas,
    betas,
    problem="nae3sat",
    **circuit_opts,
):
    """
    THEORY DOES NOT WORK. Creates a parametrized quasi-grover-mixer qaoa circuit.

    Returns:
        circ: quantum circuit
    """

    circuit_opts.setdefault("gate_opts", {})
    circuit_opts["gate_opts"].setdefault("contract", False)

    hamil = hamiltonian(G, problem)
    n = hamil.numqubit

    circ = qtn.Circuit(n, **circuit_opts)

    gates = []

    # layer of hadamards to get into plus state
    for i in range(n):
        gates.append((0, "h", i))

    circ.apply_gates(gates)

    for d in range(p):
        gates = []

        # problem Hamiltonian
        coefs, ops, qubits = hamil.gates()

        for coef, op, qubit in zip(coefs, ops, qubits):
            # circ.apply_gate_raw(op, qubit, gate_round=d)
            gates.append((d, op, coef * gammas[d], *qubit))

        # mixer Hamiltonian
        for i in range(n):
            gates.append((d, "h", i))
            gates.append((d, "x", i))

        circ.apply_gates(gates)

        qgm_gate = np.zeros((2, 2), dtype=complex)
        qgm_gate[0, 0] = 1
        qgm_gate[1, 1] = np.exp(1j * betas[d] / n)

        for i in range(n):
            circ.apply_gate_raw(qgm_gate, (i,), gate_round=d, tags="QGM")

        gates = []

        for i in range(n):
            gates.append((d, "x", i))
            gates.append((d, "h", i))

        circ.apply_gates(gates)

    return circ
