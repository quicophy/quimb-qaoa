"""
Miscellaneous utility functions.
"""

import matplotlib.pyplot as plt
import numpy as np
import quimb as qu

from .circuit import create_qaoa_circ
from .decomp import *
from .hamiltonian import hamiltonian
from .initialization import rand_ini
from .mps import create_qaoa_mps


def draw_qaoa_circ(graph, depth, qaoa_version):
    """
    Draw the QAOA circuit.

    Parameters
    ----------
    graph: ProblemGraph
        Graph representing the instance of the problem.
    depth: int
        Number of layers of gates to apply (depth 'p').
    qaoa_version: str
        Type of QAOA circuit to create.
    """

    theta_ini = rand_ini(depth)

    circ = create_qaoa_circ(
        graph, depth, theta_ini[:depth], theta_ini[depth:], qaoa_version=qaoa_version
    )

    circ.psi.draw(color=["PSI0", "H", "X", "RX", "RZ", "RZZ", "NCRZ", "QGM"])

    circ.get_rdm_lightcone_simplified(range(graph.num_nodes)).draw(
        color=["PSI0", "H", "X", "RX", "RZ", "RZZ", "NCRZ", "QGM"], show_tags=False
    )


def rehearse_qaoa_circ(
    graph,
    depth,
    qaoa_version,
    opt=None,
    backend="numpy",
    mps=False,
    draw=False,
    **ansatz_kwargs,
):
    """
    Rehearse the contraction of the QAOA circuit and compute the maximal intermediary tensor width and total contraction cost of the best contraction path.

    Parameters
    ----------
    graph: ProblemGraph
        Graph representing the instance of the problem.
    depth: int
        Number of layers of gates to apply (depth 'p').
    qaoa_version: str
        Type of QAOA circuit to create.
    opt: str
        Contraction path optimizer.
    backend: str
        Backend to use for the contraction.
    mps: bool
        If True, use MPS instead of circuit.
    draw: bool
        If True, draw the contraction width and cost.

    Returns
    -------
    width: float
        Maximal intermediary tensor width.
    cost: float
        Total contraction cost.
    local_exp_rehs: list
        List of rehearsed local expectations.
    """

    theta_ini = rand_ini(depth)

    gammas = theta_ini[:depth]
    betas = theta_ini[depth:]

    hamil = hamiltonian(graph)
    ops, qubits = hamil.operators()

    if mps:
        psi0 = create_qaoa_mps(
            graph,
            depth,
            theta_ini[:depth],
            theta_ini[depth:],
            qaoa_version=qaoa_version,
            **ansatz_kwargs,
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
            graph, depth, gammas, betas, qaoa_version, **ansatz_kwargs
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
