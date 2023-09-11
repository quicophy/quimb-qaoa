"""
Implementation of different types of mps for QAOA with the mps/mpo method.
"""

import numpy as np
import quimb as qu
import quimb.tensor as qtn
from quimb.tensor.circuit import rzz_param_gen, rx_gate_param_gen, rz_gate_param_gen
from quimb.tensor.tensor_builder import MPS_computational_state

from .hamiltonian import hamiltonian


def create_qaoa_mps(G, p, gammas, betas, qaoa_version, problem="nae3sat"):
    """
    Creates the correct qaoa mps based on user input.
    """

    if qaoa_version == "regular":
        psi = create_regular_qaoa_mps(G, p, gammas, betas, problem=problem)
    elif qaoa_version == "gm":
        psi = create_gm_qaoa_mps(G, p, gammas, betas, problem=problem)
    elif qaoa_version == "qgm":
        psi = create_qgm_qaoa_mps(G, p, gammas, betas, problem=problem)
    else:
        raise ValueError("The QAOA version is not valid.")

    return psi


def create_regular_qaoa_mps(
    G,
    p,
    gammas,
    betas,
    problem="nae3sat",
):
    """
    Creates a parametrized regular qaoa mps.

    Returns:
        circ: quantum circuit
    """

    hamil = hamiltonian(G, problem)
    n = hamil.numqubit

    # initial MPS
    psi0 = MPS_computational_state("0" * n, tags="PSI0")

    coefs, ops, qubits = hamil.gates()

    # layer of hadamards to get into plus state
    for i in range(n):
        psi0.gate_(qu.hadamard(), i, contract="swap+split", tags="H")

    for d in range(p):
        # problem Hamiltonian
        for coef, op, qubit in zip(coefs, ops, qubits):
            if op == "rzz":
                psi0.gate_with_auto_swap_(rzz_param_gen([coef * gammas[d]]), qubit)

            elif op == "rz":
                psi0.gate_(
                    rz_gate_param_gen([coef * gammas[d]]),
                    qubit,
                    contract="swap+split",
                    tags="RZ",
                )
        # print("RZZ", rzz_param_gen([-1/2*0.5]))
        # print("RZ", rz_gate_param_gen([0.5]))
        # print("RX", rx_gate_param_gen([0.5]))
        # mixer Hamiltonian
        for i in range(n):
            psi0.gate_(
                rx_gate_param_gen([2 * betas[d]]), i, contract="swap+split", tags="RX"
            )

        psi0.normalize()

    return psi0


def create_gm_qaoa_mps(
    G,
    p,
    gammas,
    betas,
    problem="nae3sat",
):
    """
    Creates a parametrized grover-mixer qaoa mps.

    Returns:
        circ: circuit
    """

    hamil = hamiltonian(G, problem)
    n = hamil.numqubit

    # initial MPS
    psi0 = MPS_computational_state("0" * n, tags="PSI0")

    # layer of hadamards to get into plus state
    for i in range(n):
        psi0.gate_(qu.hadamard(), i, contract="swap+split", tags="H")

    for d in range(p):
        # problem Hamiltonian
        coefs, ops, qubits = hamil.gates()

        for coef, op, qubit in zip(coefs, ops, qubits):
            if op == "rzz":
                psi0.gate_with_auto_swap_(rzz_param_gen([coef * gammas[d]]), qubit)

            elif op == "rz":
                psi0.gate_(
                    rz_gate_param_gen([coef * gammas[d]]),
                    qubit,
                    contract="swap+split",
                    tags="RZ",
                )

        # mixer Hamiltonian
        for i in range(n):
            psi0.gate_(qu.hadamard(), i, contract="swap+split", tags="H")
            psi0.gate_(qu.pauli("X"), i, contract="swap+split", tags="X")

        # N-Controlled RZ gate
        NCRZ = [CP()]
        for i in range(n - 2):
            NCRZ.append(ADD())
        NCRZ.append(RZ(betas[d]))

        NCRZ = qtn.tensor_1d.MatrixProductOperator(NCRZ, "udrl", tags="NCRZ")

        psi = NCRZ.apply(psi0)
        del psi0

        for i in range(n):
            psi.gate_(qu.pauli("X"), i, contract="swap+split", tags="X")
            psi.gate_(qu.hadamard(), i, contract="swap+split", tags="H")

        psi.normalize()

        psi0 = psi
        del psi

    return psi0


def create_qgm_qaoa_mps(
    G,
    p,
    gammas,
    betas,
    problem="nae3sat",
):
    """
    Creates a parametrized quasi-grover-mixer qaoa mps.

    Returns:
        circ: quantum circuit
    """

    hamil = hamiltonian(G, problem)
    n = hamil.numqubit

    # initial MPS
    psi0 = MPS_computational_state("0" * n, tags="PSI0")

    coefs, ops, qubits = hamil.gates()

    # layer of hadamards to get into plus state
    for i in range(n):
        psi0.gate_(qu.hadamard(), i, contract="swap+split", tags="H")

    for d in range(p):
        # problem Hamiltonian
        for coef, op, qubit in zip(coefs, ops, qubits):
            if op == "rzz":
                psi0.gate_with_auto_swap_(rzz_param_gen([coef * gammas[d]]), qubit)

            elif op == "rz":
                psi0.gate_(
                    rz_gate_param_gen([coef * gammas[d]]),
                    qubit,
                    contract="swap+split",
                    tags="RZ",
                )

        # mixer Hamiltonian
        for i in range(n):
            psi0.gate_(qu.hadamard(), i, contract="swap+split", tags="H")

        for coef, op, qubit in zip(coefs, ops, qubits):
            if op == "rzz":
                psi0.gate_with_auto_swap_(rzz_param_gen([-coef * betas[d]]), qubit)

            elif op == "rz":
                psi0.gate_(
                    rz_gate_param_gen([-coef * betas[d]]),
                    qubit,
                    contract="swap+split",
                    tags="RZ",
                )

        for i in range(n):
            psi0.gate_(qu.hadamard(), i, contract="swap+split", tags="H")

        psi0.normalize()

    return psi0


def create_old_qgm_qaoa_mps(
    G,
    p,
    gammas,
    betas,
    problem="nae3sat",
):
    """
    THEORY DOESN'T WORK. Creates a parametrized quasi-grover-mixer qaoa mps.

    Returns:
        circ: circuit
    """

    hamil = hamiltonian(G, problem)
    n = hamil.numqubit

    # initial MPS
    psi0 = MPS_computational_state("0" * n, tags="PSI0")

    # layer of hadamards to get into plus state
    for i in range(n):
        psi0.gate_(qu.hadamard(), i, contract="swap+split", tags="H")

    for d in range(p):
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


def CP():
    """COPY gate"""
    CP = np.zeros((2, 2, 2, 1), dtype="complex")
    CP[0, 0, 0, 0] = 1
    CP[1, 1, 1, 0] = 1
    return CP


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
