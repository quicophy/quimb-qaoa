import matplotlib.pyplot as plt

from .initialization import *
from .circuit import *
from .mps import *

def draw_qaoa_circ(G, p, qaoa_version="regular", problem="nae3sat"):
    """
    Draw the QAOA circuit.
    """

    theta_ini = rand_ini(p)

    circ = create_qaoa_circ(G, p, theta_ini[:p], theta_ini[p:], qaoa_version=qaoa_version, problem=problem)

    circ.psi.draw(color=['PSI0', 'H', 'RZZ', 'RX', 'X'])

    circ.get_rdm_lightcone_simplified(range(G.numnodes)).draw(color=['PSI0', 'H', 'RZZ', 'RX'])


def rehearse_qaoa_circ(G, p, qaoa_version="regular", problem="nae3sat", mps=False, opt=None, backend="numpy"):

    theta_ini = TQA_ini(G, p, qaoa_version=qaoa_version, problem=problem, mps=mps, opt=opt, backend=backend)

    gammas = theta_ini[:p]
    betas = theta_ini[p:]

    ops, qubits = hamiltonian_ops(G, problem=problem)

    circ = create_qaoa_circ(G, p, gammas, betas, qaoa_version=qaoa_version, problem=problem)
        
    local_exp_rehs = [
        circ.local_expectation_rehearse(op, qubit, optimize=opt, backend=backend)
            for (op, qubit) in zip(ops, qubits)
    ]

    width = []
    cost = 0
    for rehs in local_exp_rehs:
        width.append(rehs['W'])
        cost += 10**(rehs['C'])

    return width, cost

