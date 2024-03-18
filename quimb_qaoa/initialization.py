"""
Initialization methods for the initial parameters of QAOA.
"""


import numpy as np

from .contraction import compute_energy


def ini(
    graph,
    depth,
    ini_method,
    qaoa_version,
    opt=None,
    backend="numpy",
    mps=False,
    max_bond=None,
    **ansatz_opts,
):
    """
    Creates appropriate initial parameters based on user input.

    Parameters:
    -----------
    graph: ProblemGraph
        Graph representing the instance of the problem.
    depth: int
        Number of layers of gates to apply (depth 'p').
    ini_method: str
        Method to use for the initialization of the QAOA circuit
    qaoa_version: str
        Type of QAOA circuit to create.
    opt: str, optional
        Contraction path optimizer. Default is Quimb's default optimizer.
    backend: str, optional
        Backend to use for the simulation of the QAOA circuit. Default is "numpy".
    mps: bool, optional
        If True, initialize the QAOA circuit as a Matrix Product State (MPS) instead of as a general tensor networks.
    max_bond: int, optional
        Maximum bond in the contraction of the QAOA circuit. If None, the bond is not limited.

    Returns:
    --------
    theta_ini: np.ndarray
        Initial QAOA angles
    """

    if ini_method == "random":
        theta_ini = rand_ini(depth)
    elif ini_method == "tqa":
        theta_ini = TQA_ini(
            graph,
            depth,
            qaoa_version=qaoa_version,
            mps=mps,
            max_bond=max_bond,
            opt=opt,
            backend=backend,
            **ansatz_opts,
        )
    else:
        raise ValueError("The initialization method given is not valid.")

    return theta_ini


def rand_ini(depth):
    """
    Creates a list of random initial unitary parameters for the QAOA algorithm.

    Parameters:
    -----------
    depth: int
        Number of layers of gates to apply (depth 'p').

    Returns:
    --------
    theta_ini: np.ndarray
        Initial QAOA angles.
    """

    theta_ini = np.hstack(
        (
            np.random.rand(depth) * np.pi - np.pi / 2,
            np.random.rand(depth) * np.pi / 2 - np.pi / 4,
        )
    )

    return theta_ini


def TQA_ini(
    graph,
    depth,
    qaoa_version,
    opt=None,
    backend="numpy",
    mps=False,
    max_bond=None,
    **ansatz_opts,
):
    """
    Creates a list of initial unitary parameters for the QAOA algorithm. The parameters are initialized based on the Trotterized Quantum Annealing (TQA) strategy for initialization. See "Quantum Annealing Initialization of the Quantum Approximate Optimization Algorithm".

    Parameters:
    -----------
    graph: ProblemGraph
        Graph representing the instance of the problem.
    depth: int
        Number of layers of gates to apply (depth 'p').
    qaoa_version: str
        Type of QAOA circuit to create.
    problem: str
        Problem to solve.
    opt: str, optional
        Contraction path optimizer. Default is Quimb's default optimizer.
    backend: str, optional
        Backend to use for the simulation of the QAOA circuit. Default is "numpy".
    mps: bool, optional
        If True, initialize the QAOA circuit as a Matrix Product State (MPS) instead of as a general tensor networks.
    max_bond: int, optional
        Maximum bond in the contraction of the QAOA circuit. If None, the bond is not limited.

    Returns:
    --------
    theta_ini: np.ndarray
        Initial QAOA angles.
    """

    # loop over different evolution times
    times = np.linspace(1, 4, 20)

    energies = []
    for t_max in times:
        dt = t_max / depth
        t = dt * (np.arange(1, depth + 1) - 0.5)

        gamma = (t / t_max) * dt
        beta = -(1 - t / t_max) * dt

        theta = np.concatenate((gamma, beta))

        energies.append(
            compute_energy(
                theta,
                graph,
                qaoa_version=qaoa_version,
                opt=opt,
                backend=backend,
                max_bond=max_bond,
                mps=mps,
                **ansatz_opts,
            )
        )

    # find optimal time
    idx = np.argmin(energies)
    t_max = times[idx]

    dt = t_max / depth
    t = dt * (np.arange(1, depth + 1) - 0.5)

    gamma = (t / t_max) * dt
    beta = -(1 - t / t_max) * dt
    theta_ini = np.concatenate((gamma, beta))

    return theta_ini
