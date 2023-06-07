"""
Misc utility functions.
"""


import quimb.tensor as qtn
import numpy as np

from .initialization import rand_ini
from .circuit import create_qaoa_circ
from .mps import create_qaoa_mps
from .hamiltonian import hamiltonian


def draw_qaoa_circ(G, p, qaoa_version="regular", problem="nae3sat"):
    """
    Draw the QAOA circuit.
    """

    theta_ini = rand_ini(p)

    circ = create_qaoa_circ(
        G, p, theta_ini[:p], theta_ini[p:], qaoa_version=qaoa_version, problem=problem
    )

    circ.psi.draw(color=["PSI0", "H", "X", "RX", "RZZ"])

    circ.get_rdm_lightcone_simplified(range(G.numnodes)).draw(
        color=["PSI0", "H", "X", "RX", "RZZ"], show_tags=False
    )


def rehearse_qaoa_circ(
    G,
    p,
    qaoa_version="regular",
    problem="nae3sat",
    mps=False,
    opt=None,
    backend="numpy",
):
    """
    Rehearse the contraction of the QAOA circuit and compute the maximal intermediary tensor width and total contraction cost of the best contraction path.
    """

    theta_ini = rand_ini(p)

    gammas = theta_ini[:p]
    betas = theta_ini[p:]

    if mps:
        psi0 = create_qaoa_mps(
            G,
            p,
            theta_ini[:p],
            theta_ini[p:],
            qaoa_version=qaoa_version,
            problem=problem,
        )
        circ = qtn.Circuit(G.numnodes, psi0=psi0)
    else:
        circ = create_qaoa_circ(
            G, p, gammas, betas, qaoa_version=qaoa_version, problem=problem
        )

    hamil = hamiltonian(G, problem)
    ops, qubits = hamil.operators()

    local_exp_rehs = [
        circ.local_expectation_rehearse(op, qubit, optimize=opt, backend=backend)
        for (op, qubit) in zip(ops, qubits)
    ]

    width = []
    cost = 0
    for rehs in local_exp_rehs:
        width.append(rehs["W"])
        cost += 10 ** (rehs["C"])

    return min(width), np.log10(cost)
