"""
Implementation of different types of Matrix Product States (MPS) for QAOA with the MPS/MPO method.
"""

import numpy as np
import quimb as qu
import quimb.tensor as qtn
from quimb.tensor.circuit import rx_gate_param_gen, rz_gate_param_gen, rzz_param_gen
from quimb.tensor.tensor_builder import MPS_computational_state

from .hamiltonian import hamiltonian


def create_qaoa_mps(graph, depth, gammas, betas, qaoa_version, assumptions=[]):
    """
    Creates an appropriate QAOA MPS based on user input.

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
        Type of QAOA MPS to create.

    Returns
    -------
    psi: MatrixProductState
        The QAOA MPS.
    """

    if qaoa_version == "regular":
        psi = create_reg_qaoa_mps(graph, depth, gammas, betas)
    elif qaoa_version == "grover-mixer":
        psi = create_gm_qaoa_mps(graph, depth, gammas, betas)
    elif qaoa_version == "vqcount-regular":
        psi = create_vqcount_reg_qaoa_mps(
            graph, depth, gammas, betas, assumptions=assumptions
        )
    elif qaoa_version == "vqcount-grover-mixer":
        psi = create_vqcount_gm_qaoa_mps(
            graph, depth, gammas, betas, assumptions=assumptions
        )
    # elif qaoa_version == "tdvp":
    #     psi = _create_tdvp_qaoa_mps(graph, depth, gammas, betas)
    else:
        raise ValueError("The QAOA version given is not valid.")

    return psi


def create_reg_qaoa_mps(
    graph,
    depth,
    gammas,
    betas,
):
    """
    Creates the original QAOA MPS, i.e. with the X-mixer.

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

    Returns
    -------
    psi: MatrixProductState
        The QAOA MPS.
    """

    hamil = hamiltonian(graph)
    n = hamil.numqubit  # may differ from number of nodes

    # initial MPS
    psi0 = MPS_computational_state("0" * n, tags="PSI0")

    coefs, ops, qubits = hamil.gates()

    # layer of hadamards to get into plus state
    for i in range(n):
        psi0.gate_(qu.hadamard(), i, contract="swap+split", tags="H")

    for p in range(depth):
        # problem Hamiltonian
        for coef, op, qubit in zip(coefs, ops, qubits):
            if op == "rzz":
                psi0.gate_with_auto_swap_(rzz_param_gen([coef * gammas[p]]), qubit)

            elif op == "rz":
                psi0.gate_(
                    rz_gate_param_gen([coef * gammas[p]]),
                    qubit,
                    contract="swap+split",
                    tags="RZ",
                )

        # mixer Hamiltonian
        for i in range(n):
            psi0.gate_(
                rx_gate_param_gen([2 * betas[p]]), i, contract="swap+split", tags="RX"
            )

        psi0.normalize()

    return psi0


def create_gm_qaoa_mps(
    graph,
    depth,
    gammas,
    betas,
):
    """
    Creates the Grover-Mixer QAOA (GM-QAOA) MPS.

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

    Returns
    -------
    psi: MatrixProductState
        The QAOA MPS.
    """

    hamil = hamiltonian(graph)
    n = hamil.numqubit  # may differ from number of nodes

    # initial MPS
    psi0 = MPS_computational_state("0" * n, tags="PSI0")

    # layer of hadamards to get into plus state
    for i in range(n):
        psi0.gate_(qu.hadamard(), i, contract="swap+split", tags="H")

    for p in range(depth):
        # problem Hamiltonian
        coefs, ops, qubits = hamil.gates()

        for coef, op, qubit in zip(coefs, ops, qubits):
            if op == "rzz":
                psi0.gate_with_auto_swap_(rzz_param_gen([coef * gammas[p]]), qubit)

            elif op == "rz":
                psi0.gate_(
                    rz_gate_param_gen([coef * gammas[p]]),
                    qubit,
                    contract="swap+split",
                    tags="RZ",
                )

        # mixer Hamiltonian
        for i in range(n):
            psi0.gate_(qu.hadamard(), i, contract="swap+split", tags="H")
            psi0.gate_(qu.pauli("X"), i, contract="swap+split", tags="X")

        # multi-control phase-shift gate
        ncrz_gate = [CP_TOP()]
        for i in range(n - 2):
            ncrz_gate.append(ADD())
        ncrz_gate.append(RZ(2 * betas[p]))

        ncrz_gate = qtn.tensor_1d.MatrixProductOperator(ncrz_gate, "udrl", tags="NCRZ")

        psi = ncrz_gate.apply(psi0)
        del psi0

        for i in range(n):
            psi.gate_(qu.pauli("X"), i, contract="swap+split", tags="X")
            psi.gate_(qu.hadamard(), i, contract="swap+split", tags="H")

        psi.normalize()

        psi0 = psi
        del psi

    return psi0


def create_vqcount_reg_qaoa_mps(
    graph,
    depth,
    gammas,
    betas,
    assumptions,
):
    """
    Creates the VQCount original QAOA MPS, i.e. with the X-mixer.

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
    assumptions: iterable of str
        The qubit to fixed in the QAOA circuit for the VQCount algorithm.

    Returns
    -------
    psi: MatrixProductState
        The QAOA MPS.
    """

    hamil = hamiltonian(graph)
    n = hamil.numqubit  # may differ from number of nodes

    # initial MPS
    psi0 = MPS_computational_state("0" * n, tags="PSI0")

    coefs, ops, qubits = hamil.gates()

    for i, var in enumerate(assumptions):
        if var == str(1):
            psi0.gate_(qu.pauli("X"), i, contract="swap+split", tags="X")

    # layer of hadamards to get into plus state
    for i in range(len(assumptions), n):
        psi0.gate_(qu.hadamard(), i, contract="swap+split", tags="H")

    for p in range(depth):
        # problem Hamiltonian
        for coef, op, qubit in zip(coefs, ops, qubits):
            if op == "rzz":
                psi0.gate_with_auto_swap_(rzz_param_gen([coef * gammas[p]]), qubit)

            elif op == "rz":
                psi0.gate_(
                    rz_gate_param_gen([coef * gammas[p]]),
                    qubit,
                    contract="swap+split",
                    tags="RZ",
                )

        # mixer Hamiltonian
        for i in range(len(assumptions), n):
            psi0.gate_(
                rx_gate_param_gen([2 * betas[p]]), i, contract="swap+split", tags="RX"
            )

        psi0.normalize()

    return psi0


def create_vqcount_gm_qaoa_mps(
    graph,
    depth,
    gammas,
    betas,
    assumptions,
):
    """
    Creates the VQCount Grover-Mixer QAOA (GM-QAOA) MPS.

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
    assumptions: iterable of str
        The qubit to fixed in the QAOA circuit for the VQCount algorithm.

    Returns
    -------
    psi: MatrixProductState
        The QAOA MPS.
    """

    hamil = hamiltonian(graph)
    n = hamil.numqubit  # may differ from number of nodes

    # initial MPS
    psi0 = MPS_computational_state("0" * n, tags="PSI0")

    for i, var in enumerate(assumptions):
        if var == str(1):
            psi0.gate_(qu.pauli("X"), i, contract="swap+split", tags="X")

    # layer of hadamards to get into plus state
    for i in range(len(assumptions), n):
        psi0.gate_(qu.hadamard(), i, contract="swap+split", tags="H")

    for p in range(depth):
        # problem Hamiltonian
        coefs, ops, qubits = hamil.gates()

        for coef, op, qubit in zip(coefs, ops, qubits):
            if op == "rzz":
                psi0.gate_with_auto_swap_(rzz_param_gen([coef * gammas[p]]), qubit)

            elif op == "rz":
                psi0.gate_(
                    rz_gate_param_gen([coef * gammas[p]]),
                    qubit,
                    contract="swap+split",
                    tags="RZ",
                )

        # mixer Hamiltonian
        for i in range(len(assumptions), n):
            psi0.gate_(qu.hadamard(), i, contract="swap+split", tags="H")
            psi0.gate_(qu.pauli("X"), i, contract="swap+split", tags="X")

        ncrz_gate = []

        # multi-control phase-shift gate
        if len(assumptions) != 0:
            ncrz_gate.append(ID_TOP())

            for i in range(len(assumptions) - 1):
                ncrz_gate.append(ID_MID())

            if n - len(assumptions) > 1:
                ncrz_gate.append(CP_MID())

        else:
            ncrz_gate.append(CP_TOP())

        for i in range(n - len(assumptions) - 2):
            ncrz_gate.append(ADD())

        ncrz_gate.append(RZ(2 * betas[p]))

        ncrz_gate = qtn.tensor_1d.MatrixProductOperator(ncrz_gate, "udrl", tags="NCRZ")

        psi = ncrz_gate.apply(psi0)
        del psi0

        for i in range(len(assumptions), n):
            psi.gate_(qu.pauli("X"), i, contract="swap+split", tags="X")
            psi.gate_(qu.hadamard(), i, contract="swap+split", tags="H")

        psi.normalize()

        psi0 = psi
        del psi

    return psi0


def _create_tdvp_qaoa_mps(
    graph,
    depth,
    gammas,
    betas,
):
    """
    UNTESTED. Creates the Time-Dependent Variational Principal (TDVP) QAOA MPS.

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

    Returns
    -------
    psi: MatrixProductState
        The QAOA MPS.
    """

    return


def _create_qgm1_qaoa_mps(
    graph,
    depth,
    gammas,
    betas,
):
    """
    UNTESTED. Creates the Quasi-Grover-Mixer QAOA (QGM-QAOA) MPS.

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

    Returns
    -------
    psi: MatrixProductState
        The QAOA MPS.
    """

    hamil = hamiltonian(graph)
    n = hamil.numqubit  # may differ from number of nodes

    # initial MPS
    psi0 = MPS_computational_state("0" * n, tags="PSI0")

    coefs, ops, qubits = hamil.gates()

    # layer of hadamards to get into plus state
    for i in range(n):
        psi0.gate_(qu.hadamard(), i, contract="swap+split", tags="H")

    for p in range(depth):
        # problem Hamiltonian
        for coef, op, qubit in zip(coefs, ops, qubits):
            if op == "rzz":
                psi0.gate_with_auto_swap_(rzz_param_gen([coef * gammas[p]]), qubit)

            elif op == "rz":
                psi0.gate_(
                    rz_gate_param_gen([coef * gammas[p]]),
                    qubit,
                    contract="swap+split",
                    tags="RZ",
                )

        # mixer Hamiltonian
        for i in range(n):
            psi0.gate_(qu.hadamard(), i, contract="swap+split", tags="H")

        for coef, op, qubit in zip(coefs, ops, qubits):
            if op == "rzz":
                psi0.gate_with_auto_swap_(rzz_param_gen([-coef * betas[p]]), qubit)

            elif op == "rz":
                psi0.gate_(
                    rz_gate_param_gen([-coef * betas[p]]),
                    qubit,
                    contract="swap+split",
                    tags="RZ",
                )

        for i in range(n):
            psi0.gate_(qu.hadamard(), i, contract="swap+split", tags="H")

        psi0.normalize()

    return psi0


def _create_qgm2_qaoa_mps(
    graph,
    depth,
    gammas,
    betas,
):
    """
    UNTESTED. Creates the Quasi-Grover-Mixer QAOA (QGM-QAOA) MPS.

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

    Returns
    -------
    psi: MatrixProductState
        The QAOA MPS.
    """

    hamil = hamiltonian(graph)
    n = hamil.numqubit  # may differ from number of nodes

    # initial MPS
    psi0 = MPS_computational_state("0" * n, tags="PSI0")

    # layer of hadamards to get into plus state
    for i in range(n):
        psi0.gate_(qu.hadamard(), i, contract="swap+split", tags="H")

    for d in range(depth):
        # problem Hamiltonian
        coefs, ops, qubits = hamil.gates()

        for coef, op, qubit in zip(coefs, ops, qubits):
            if op == "rzz":
                psi0.gate_with_auto_swap_(rzz_param_gen([coef * gammas[d]]), qubit)

            elif op == "rz":
                psi0.gate_(
                    qu.phase_gate(coef * gammas[d]),
                    qubit,
                    contract="swap+split",
                    tags="RZ",
                )

        # mixer Hamiltonian
        for i in range(n):
            psi0.gate_(qu.hadamard(), i, contract="swap+split", tags="H")
            psi0.gate_(qu.pauli("X"), i, contract="swap+split", tags="X")

        # quasi-grover-mixer gate
        qgm_gate = np.zeros((2, 2), dtype=complex)
        qgm_gate[0, 0] = 1
        qgm_gate[1, 1] = np.exp(1j * betas[d] / n)

        for i in range(n):
            psi0.gate_(qgm_gate, i, contract="swap+split", tags="QGM")

        for i in range(n):
            psi0.gate_(qu.pauli("X"), i, contract="swap+split", tags="X")
            psi0.gate_(qu.hadamard(), i, contract="swap+split", tags="H")

        psi0.normalize()

    return psi0


def RZ(beta):
    "Z-Rotation gate"
    RZ = np.zeros((2, 2, 1, 2), dtype="complex")
    RZ[0, 0, 0, 0] = 1
    RZ[1, 1, 0, 0] = 1
    RZ[0, 0, 0, 1] = 1
    RZ[1, 1, 0, 1] = np.exp(-1j * beta)
    return RZ


def ADD():
    """ADD gate"""
    ADD = np.zeros((2, 2, 2, 2), dtype="complex")
    ADD[0, 0, 0, 0] = 1
    ADD[0, 0, 1, 0] = 0
    ADD[0, 0, 1, 1] = 0
    ADD[0, 0, 0, 1] = 1
    ADD[1, 1, 0, 0] = 1
    ADD[1, 1, 1, 0] = 0
    ADD[1, 1, 0, 1] = 0
    ADD[1, 1, 1, 1] = 1
    return ADD


def CP_TOP():
    """COPY gate"""
    CP = np.zeros((2, 2, 2, 1), dtype="complex")
    CP[0, 0, 0, 0] = 1
    CP[1, 1, 1, 0] = 1
    return CP


def CP_MID():
    """COPY gate"""
    CP = np.zeros((2, 2, 2, 2), dtype="complex")
    CP[0, 0, 0, 0] = 1
    CP[1, 1, 1, 0] = 1
    return CP


def ID_TOP():
    ID = np.zeros((2, 2, 2, 1), dtype="complex")
    ID[0, 0, 0, 0] = 1
    ID[1, 1, 0, 0] = 1
    return ID


def ID_MID():
    ID = np.zeros((2, 2, 2, 2), dtype="complex")
    ID[0, 0, 0, 0] = 1
    ID[1, 1, 0, 0] = 1
    return ID
