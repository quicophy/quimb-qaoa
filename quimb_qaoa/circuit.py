"""
Implementation of different types of circuits for QAOA with general tensor networks.
"""


import numpy as np
import quimb as qu
import quimb.tensor as qtn

from .hamiltonian import hamiltonian


def create_qaoa_circ(graph, depth, gammas, betas, qaoa_version, **circuit_opts):
    """
    Creates an appropriate QAOA circuit based on user input.

    Parameters
    ----------
    graph: ProblemGraph
        Graph representing the instance of the problem.
    depth: int
        Number of layers of gates to apply (depth 'p').
    gammas: iterable of floats
        Interaction angles (problem Hamiltonian) for each layer.
    betas: iterable of floats
        Rotation angles (mixer Hamiltonian) for each layer.
    qaoa_version: str
        The type of QAOA circuit to create.
    circuit_opts: dict
        Supplied to :class:`~quimb.tensor.circuit.Circuit`.

    Returns
    -------
    circ: Circuit
        The QAOA circuit.
    """

    if qaoa_version == "regular":
        circ = create_reg_qaoa_circ(graph, depth, gammas, betas, **circuit_opts)
    elif qaoa_version == "grover-mixer":
        circ = create_gm_qaoa_circ(graph, depth, gammas, betas, **circuit_opts)
    else:
        raise ValueError("The QAOA version given is not valid.")

    return circ


def create_reg_qaoa_circ(
    graph,
    depth,
    gammas,
    betas,
    **circuit_opts,
):
    """
    Creates the original QAOA circuit, i.e. with the X-mixer.

    Parameters
    ----------
    graph: ProblemGraph
        Graph representing the instance of the problem.
    depth: int
        Number of layers of gates to apply (depth 'p').
    gammas: iterable of floats
        Interaction angles (problem Hamiltonian) for each layer.
    betas: iterable of floats
        Rotation angles (mixer Hamiltonian) for each layer.
    circuit_opts: dict
        Supplied to :class:`~quimb.tensor.circuit.Circuit`.

    Returns
    -------
    circ: Circuit
        The QAOA circuit.
    """

    # set by default since the RZZ gate is diagonal
    circuit_opts.setdefault("gate_opts", {})
    circuit_opts["gate_opts"].setdefault("contract", False)

    hamil = hamiltonian(graph)
    n = hamil.numqubit  # may differ from the number of nodes

    circ = qtn.Circuit(n, **circuit_opts)

    gates = []

    # layer of hadamards to get into plus state
    for i in range(n):
        gates.append((0, "h", i))

    circ.apply_gates(gates)

    for p in range(depth):
        gates = []

        # problem Hamiltonian
        coefs, ops, qubits = hamil.gates()

        for coef, op, qubit in zip(coefs, ops, qubits):
            gates.append((p, op, coef * gammas[p], *qubit))

        # mixer Hamiltonian
        for i in range(n):
            gates.append((p, "rx", 2 * betas[p], i))

        circ.apply_gates(gates)

    return circ


def create_mod_reg_qaoa_circ(
    graph,
    depth,
    gammas,
    betas,
    assumptions,
    **circuit_opts,
):
    """
    Creates the original QAOA circuit, i.e. with the X-mixer.

    Parameters
    ----------
    graph: ProblemGraph
        Graph representing the instance of the problem.
    depth: int
        Number of layers of gates to apply (depth 'p').
    gammas: iterable of floats
        Interaction angles (problem Hamiltonian) for each layer.
    betas: iterable of floats
        Rotation angles (mixer Hamiltonian) for each layer.
    circuit_opts: dict
        Supplied to :class:`~quimb.tensor.circuit.Circuit`.

    Returns
    -------
    circ: Circuit
        The QAOA circuit.
    """

    # set by default since the RZZ gate is diagonal
    circuit_opts.setdefault("gate_opts", {})
    circuit_opts["gate_opts"].setdefault("contract", False)

    hamil = hamiltonian(graph)
    n = hamil.numqubit  # may differ from the number of nodes

    circ = qtn.Circuit(n, **circuit_opts)

    gates = []
    # layer of hadamards to get into plus state
    for i in range(len(assumptions), n):
        gates.append((0, "h", i))

    for i, var in enumerate(assumptions):
        if var == str(1):
            gates.append((0, "X", i))

    circ.apply_gates(gates)

    for p in range(depth):
        gates = []

        # problem Hamiltonian
        coefs, ops, qubits = hamil.gates()

        for coef, op, qubit in zip(coefs, ops, qubits):
            gates.append((p, op, coef * gammas[p], *qubit))

        # mixer Hamiltonian
        for i in range(len(assumptions), n):
            gates.append((p, "rx", 2 * betas[p], i))

        circ.apply_gates(gates)

    return circ


def create_gm_qaoa_circ(
    graph,
    depth,
    gammas,
    betas,
    **circuit_opts,
):
    """
    ONLY WORKS UP TO AROUND 14 QUBITS DUE TO N-CONTROL GATE. USE THE MPS VERSION FOR HIGHER NUMBER OF QUBITS. Creates the Grover-Mixer QAOA (GM-QAOA) circuit.

    Parameters
    ----------
    graph: ProblemGraph
        Graph representing the instance of the problem.
    depth: int
        Number of layers of gates to apply (depth 'p').
    gammas: iterable of floats
        Interaction angles (problem Hamiltonian) for each layer.
    betas: iterable of floats
        Rotation angles (mixer Hamiltonian) for each layer.
    circuit_opts: dict
        Supplied to :class:`~quimb.tensor.circuit.Circuit`.

    Returns
    -------
    circ: Circuit
        The QAOA circuit.
    """

    # set by default since the RZZ gate is diagonal
    circuit_opts.setdefault("gate_opts", {})
    circuit_opts["gate_opts"].setdefault("contract", False)

    hamil = hamiltonian(graph)
    n = hamil.numqubit  # may differ from the number of nodes

    circ = qtn.Circuit(n, **circuit_opts)

    gates = []

    # layer of hadamards to get into plus state
    for i in range(n):
        gates.append((0, "h", i))

    circ.apply_gates(gates)

    for p in range(depth):
        gates = []

        # problem Hamiltonian
        coefs, ops, qubits = hamil.gates()
        for coef, op, qubit in zip(coefs, ops, qubits):
            gates.append((p, op, coef * gammas[p], *qubit))

        # mixer Hamiltonian
        for i in range(n):
            gates.append((p, "h", i))
            gates.append((p, "x", i))

        circ.apply_gates(gates)

        # multi-control phase-shift gate
        ncrz_gate = np.eye(2**n, dtype=complex)
        ncrz_gate[-1, -1] = np.exp(-2j * betas[p])
        circ.apply_gate_raw(ncrz_gate, range(0, n), gate_round=p, tags="NCRZ")

        gates = []

        for i in range(n):
            gates.append((p, "x", i))
            gates.append((p, "h", i))

        circ.apply_gates(gates)

    return circ


def _create_xy_qaoa_circ(
    graph,
    depth,
    gammas,
    betas,
    **circuit_opts,
):
    """
    UNTESTED. Creates the XY-mixer QAOA circuit.

    Parameters
    ----------
    graph: ProblemGraph
        Graph representing the instance of the problem.
    depth: int
        Number of layers of gates to apply (depth 'p').
    gammas: iterable of floats
        Interaction angles (problem Hamiltonian) for each layer.
    betas: iterable of floats
        Rotation angles (mixer Hamiltonian) for each layer.
    circuit_opts: dict
        Supplied to :class:`~quimb.tensor.circuit.Circuit`.

    Returns
    -------
    circ: Circuit
        The QAOA circuit.
    """

    circuit_opts.setdefault("gate_opts", {})
    circuit_opts["gate_opts"].setdefault("contract", False)

    hamil = hamiltonian(graph)
    n = hamil.numqubit

    dims = n * [2]

    wstate = qu.gen.states.w_state(n)

    mps = qtn.tensor_1d.MatrixProductState.from_dense(wstate, dims)

    circ = qtn.Circuit(n, psi0=mps, **circuit_opts)

    for p in range(depth):
        gates = []
        # problem Hamiltonian
        coefs, ops, qubits = hamil.gates()

        for coef, op, qubit in zip(coefs, ops, qubits):
            # circ.apply_gate_raw(op, qubit, gate_round=d)
            gates.append((p, op, coef * gammas[p], *qubit))

        circ.apply_gates(gates)
        # mixer Hamiltonian

        for b in circ.sample(18):
            print(b)
        sn = int(n**1 / 2)
        for j in range(sn):
            rxy = np.array(
                [
                    [1, 0, 0, 0],
                    [0, -1j * np.sin(-betas[p]), np.cos(betas[p]), 0],
                    [0, np.cos(betas[p]), -1j * np.sin(-betas[p]), 0],
                    [0, 0, 0, 1],
                ]
            )
            v = (j + 1) % n
            circ.apply_gate_raw(rxy, (j, v), tags="iswap")

    return circ


def _create_qgm1_qaoa_circ(
    graph,
    depth,
    gammas,
    betas,
    **circuit_opts,
):
    """
    UNTESTED. Creates the Quasi-Grover-Mixer QAOA circuit.

    Parameters
    ----------
    graph: ProblemGraph
        Graph representing the instance of the problem.
    depth: int
        Number of layers of gates to apply (depth 'p').
    gammas: iterable of floats
        Interaction angles (problem Hamiltonian) for each layer.
    betas: iterable of floats
        Rotation angles (mixer Hamiltonian) for each layer.
    circuit_opts: dict
        Supplied to :class:`~quimb.tensor.circuit.Circuit`.

    Returns
    -------
    circ: Circuit
        The QAOA circuit.
    """

    circuit_opts.setdefault("gate_opts", {})
    circuit_opts["gate_opts"].setdefault("contract", False)

    hamil = hamiltonian(graph)
    n = hamil.numqubit

    circ = qtn.Circuit(n, **circuit_opts)

    gates = []

    # layer of hadamards to get into plus state
    for i in range(n):
        gates.append((0, "h", i))

    circ.apply_gates(gates)

    for p in range(depth):
        gates = []

        # problem Hamiltonian
        coefs, ops, qubits = hamil.gates()

        for coef, op, qubit in zip(coefs, ops, qubits):
            # circ.apply_gate_raw(op, qubit, gate_round=d)
            gates.append((p, op, coef * gammas[p], *qubit))

        # mixer Hamiltonian
        for i in range(n):
            gates.append((p, "h", i))
            gates.append((p, "x", i))

        circ.apply_gates(gates)

        # quasi-grover-mixer gate
        qgm_gate = np.zeros((2, 2), dtype=complex)
        qgm_gate[0, 0] = 1
        qgm_gate[1, 1] = np.exp(1j * betas[p] / n)

        for i in range(n):
            circ.apply_gate_raw(qgm_gate, (i,), gate_round=p, tags="QGM")

        gates = []

        for i in range(n):
            gates.append((p, "x", i))
            gates.append((p, "h", i))

        circ.apply_gates(gates)

    return circ


def _create_qgm2_qaoa_circ(
    graph,
    depth,
    gammas,
    betas,
    **circuit_opts,
):
    """
    UNTESTED. Creates the Quasi-Grover-Mixer QAOA circuit.

    Parameters
    ----------
    graph: ProblemGraph
        Graph representing the instance of the problem.
    depth: int
        Number of layers of gates to apply (depth 'p').
    gammas: iterable of floats
        Interaction angles (problem Hamiltonian) for each layer.
    betas: iterable of floats
        Rotation angles (mixer Hamiltonian) for each layer.
    circuit_opts: dict
        Supplied to :class:`~quimb.tensor.circuit.Circuit`.

    Returns
    -------
    circ: Circuit
        The QAOA circuit.
    """

    circuit_opts.setdefault("gate_opts", {})
    circuit_opts["gate_opts"].setdefault("contract", False)

    hamil = hamiltonian(graph)
    n = hamil.numqubit

    circ = qtn.Circuit(n, **circuit_opts)

    gates = []

    # layer of hadamards to get into plus state
    for i in range(n):
        gates.append((0, "h", i))

    circ.apply_gates(gates)

    for p in range(depth):
        gates = []

        # problem Hamiltonian
        coefs, ops, qubits = hamil.gates()

        for coef, op, qubit in zip(coefs, ops, qubits):
            # circ.apply_gate_raw(op, qubit, gate_round=d)
            gates.append((p, op, coef * gammas[p], *qubit))

        # mixer Hamiltonian
        for i in range(n):
            gates.append((p, "h", i))

        for coef, op, qubit in zip(coefs, ops, qubits):
            # circ.apply_gate_raw(op, qubit, gate_round=d)
            gates.append((p, op, -coef * betas[p], *qubit))

        for i in range(n):
            gates.append((p, "h", i))

        circ.apply_gates(gates)

    return circ
