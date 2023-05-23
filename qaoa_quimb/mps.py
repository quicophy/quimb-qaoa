"""
Implementation of different types of mps for QAOA with the mps/mpo method.
"""


import quimb.tensor as qtn
from quimb.tensor.tensor_builder import MPS_computational_state

from .hamiltonian import hamiltonian_gates


def create_qaoa_mps(G, p, gammas, betas, qaoa_version, problem="nae3sat"):
    """
    Creates the correct qaoa mps based on user input.
    """

    if qaoa_version == 'regular':
        psi = create_regular_qaoa_mps(G, p, gammas, betas, problem=problem)
    elif qaoa_version == 'gm':
        psi = create_gm_qaoa_mps(G, p, gammas, betas, problem=problem)
    else:
        raise ValueError('The QAOA version is not valid.')

    return psi

def create_regular_qaoa_mps(G, p, gammas, betas, problem="nae3sat", **circuit_opts,):
        """
        Creates a parametrized regular qaoa mps.
        
        Returns:
            circ: quantum circuit
        """

        n = G.numnodes

        # initial MPS
        psi0 = MPS_computational_state('0'*n, tags="PSI0")

        # layer of hadamards to get into plus state
        for i in range(n):
            psi0.gate_(H(), i, contract="swap+split", tags="H")

        for d in range(p):

            # problem Hamiltonian
            coefs, ops, qubits = hamiltonian_gates(G, problem=problem)

            for (coef, op, qubit) in zip(coefs, ops, qubits):

                if op == "rzz":
            
                    psi0.gate_with_auto_swap_(coef*RZZ(gammas[d]), qubit)

                elif op == "rz":

                    psi0.gate_with_auto_swap_(coef*RZ(gammas[d]), qubit)

            # mixer Hamiltonian
            for i in range(n):
                 psi0.gate_(RX(-2*betas[d]), i, contract="swap+split", tags="RX")

        return psi0
    
def create_gm_qaoa_mps(G, p, gammas, betas, problem="nae3sat", **circuit_opts,):
    """
    Creates a parametrized grover-mixer qaoa mps.

    Returns:
        circ: circuit
    """

    n = G.numnodes

    # initial MPS
    psi0 = MPS_computational_state('0'*n, tags="PSI0")

    # layer of hadamards to get into plus state
    for i in range(n):
        psi0.gate_(H(), i, contract="swap+split", tags="H")

    for d in range(p):

        # problem Hamiltonian
        coefs, ops, qubits = hamiltonian_gates(G, problem=problem)

        for (coef, op, qubit) in zip(coefs, ops, qubits):

            if op == "rzz":
        
                psi0.gate_with_auto_swap_(coef*RZZ(gammas[d]), qubit)

            elif op == "rz":

                psi0.gate_with_auto_swap_(coef*RZ(gammas[d]), qubit)

        # mixer Hamiltonian
        for i in range(n):
            psi0.gate_(H(), i, contract="swap+split", tags="H")
            psi0.gate_(X(), i, contract="swap+split", tags="X")

        # N-Controlled RZ gate
        NCRZ = [CP()]
        for i in range(n-2):
            NCRZ.append(ADD())
        NCRZ.append(RZ(betas[d]))

        NCRZ = qtn.tensor_1d.MatrixProductOperator(NCRZ, 'udrl', tags="NCRZ")

        psi = NCRZ.apply(psi0)
        del psi0

        psi.normalize()

        for i in range(n):
            psi.gate_(X(), i, contract="swap+split", tags="X")
            psi.gate_(H(), i, contract="swap+split", tags="H")

        psi0 = psi
        del psi            

    return psi0