"""
Misc utility functions.
"""

import quimb as qu
import numpy as np
import matplotlib.pyplot as plt

from .initialization import rand_ini
from .circuit import create_qaoa_circ
from .mps import create_qaoa_mps
from .hamiltonian import hamiltonian
from .decomp import *


def draw_qaoa_circ(G, p, qaoa_version="regular", problem="nae3sat"):
    """
    Draw the QAOA circuit.
    """

    theta_ini = rand_ini(p)

    circ = create_qaoa_circ(
        G, p, theta_ini[:p], theta_ini[p:], qaoa_version=qaoa_version, problem=problem
    )

    circ.psi.draw(color=["PSI0", "H", "X", "RX", "RZ", "RZZ"])

    circ.get_rdm_lightcone_simplified(range(G.numnodes)).draw(
        color=["PSI0", "H", "X", "RX", "RZ", "RZZ"], show_tags=False
    )


def rehearse_qaoa_circ(
    G,
    p,
    qaoa_version="regular",
    problem="nae3sat",
    mps=False,
    opt=None,
    backend="numpy",
    draw=False,
):
    """
    Rehearse the contraction of the QAOA circuit and compute the maximal intermediary tensor width and total contraction cost of the best contraction path.
    """

    theta_ini = rand_ini(p)

    gammas = theta_ini[:p]
    betas = theta_ini[p:]

    hamil = hamiltonian(G, problem)
    ops, qubits = hamil.operators()

    if mps:
        psi0 = create_qaoa_mps(
            G,
            p,
            theta_ini[:p],
            theta_ini[p:],
            qaoa_version=qaoa_version,
            problem=problem,
        )

        local_exp_rehs = [
            psi0.local_expectation_exact(
                op, qubit, optimize=opt, backend=backend, rehearse=True
            )
            for (op, qubit) in zip(ops, qubits)
        ]

        width = []
        cost = 0
        for rehs in local_exp_rehs:
            width.append(np.log2(int(rehs.largest_intermediate)))
            cost += rehs.opt_cost / 2

        if draw:
            with plt.style.context(qu.NEUTRAL_STYLE):
                fig, ax1 = plt.subplots()
                ax1.plot(
                    [
                        np.log2(int(rehs.largest_intermediate))
                        for rehs in local_exp_rehs
                    ],
                    color="green",
                )
                ax1.set_ylabel("contraction width, $W$, [log2]", color="green")
                ax1.tick_params(axis="y", labelcolor="green")

                ax2 = ax1.twinx()
                ax2.plot(
                    [np.log10(rehs.opt_cost / 2) for rehs in local_exp_rehs],
                    color="orange",
                )
                ax2.set_ylabel("contraction cost, $C$, [log10]", color="orange")
                ax2.tick_params(axis="y", labelcolor="orange")
                plt.show()

    else:
        circ = create_qaoa_circ(
            G, p, gammas, betas, qaoa_version=qaoa_version, problem=problem
        )

        local_exp_rehs = [
            circ.local_expectation(
                op, qubit, optimize=opt, backend=backend, rehearse=True
            )
            for (op, qubit) in zip(ops, qubits)
        ]

        width = []
        cost = 0
        for rehs in local_exp_rehs:
            width.append(rehs["W"])
            cost += 10 ** (rehs["C"])

        if draw:
            with plt.style.context(qu.NEUTRAL_STYLE):
                fig, ax1 = plt.subplots()
                ax1.plot([rehs["W"] for rehs in local_exp_rehs], color="green")
                ax1.set_ylabel("contraction width, $W$, [log2]", color="green")
                ax1.tick_params(axis="y", labelcolor="green")

                ax2 = ax1.twinx()
                ax2.plot([rehs["C"] for rehs in local_exp_rehs], color="orange")
                ax2.set_ylabel("contraction cost, $C$, [log10]", color="orange")
                ax2.tick_params(axis="y", labelcolor="orange")
                plt.show()

    return max(width), np.log10(cost), local_exp_rehs
