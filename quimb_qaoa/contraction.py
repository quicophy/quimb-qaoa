"""
Functions for the contraction of QAOA.
"""

import numpy as np
from scipy.optimize import minimize

from .circuit import create_qaoa_circ
from .hamiltonian import hamiltonian
from .instantiation import instantiate_ansatz
from .mps import create_qaoa_mps


def compute_energy(
    theta,
    graph,
    qaoa_version,
    opt=None,
    backend="numpy",
    mps=False,
    max_bond=None,
    **ansatz_opts,
):
    """
    Computes the energy value of a QAOA circuit based on user input.

    Parameters:
    -----------
    theta: np.ndarray
        QAOA angles of the form (gamma_1, ..., gamma_p, beta_1, ..., beta_p).
    graph: ProblemGraph
        Graph representing the instance of the problem.
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
    energy: float
        Energy value of the QAOA circuit.
    """
    if max_bond is None:
        energy = compute_exact_energy(
            theta,
            graph,
            qaoa_version,
            opt=opt,
            backend=backend,
            mps=mps,
            **ansatz_opts,
        )
    else:
        # energy = _compute_approx_energy(
        #     theta,
        #     graph,
        #     qaoa_version,
        #     max_bond,
        #     opt=opt,
        #     backend=backend,
        #     mps=mps,
        # )
        raise ValueError("Approximate energy computation not yet implemented.")

    return energy


def compute_exact_energy(
    theta,
    graph,
    qaoa_version,
    opt=None,
    backend="numpy",
    mps=False,
    **ansatz_opts,
):
    """
    Find the expectation value of the problem Hamiltonian with given QAOA angles.

    Parameters:
    -----------
    theta: np.ndarray
        QAOA angles of the form (gamma_1, ..., gamma_p, beta_1, ..., beta_p).
    graph: ProblemGraph
        Graph representing the instance of the problem.
    qaoa_version: str
        Type of QAOA circuit to create.
    opt: str, optional
        Contraction path optimizer. Default is Quimb's default optimizer.
    backend: str, optional
        Backend to use for the simulation of the QAOA circuit. Default is "numpy".
    mps: bool, optional
        If True, initialize the QAOA circuit as a Matrix Product State (MPS) instead of as a general tensor networks.

    Returns:
    --------
    energy: float
        Expectation value of the problem Hamiltonian with the given QAOA angles.
    """

    depth = len(theta) // 2
    gammas = theta[:depth]
    betas = theta[depth:]

    hamil = hamiltonian(graph)
    ops, qubits = hamil.operators()

    if mps:
        psi = create_qaoa_mps(graph, depth, gammas, betas, qaoa_version, **ansatz_opts)
        ens = [
            psi.local_expectation_exact(op, qubit, optimize=opt, backend=backend)
            for (op, qubit) in zip(ops, qubits)
        ]

    else:
        circ = create_qaoa_circ(
            graph, depth, gammas, betas, qaoa_version, **ansatz_opts
        )
        ens = [
            circ.local_expectation(op, qubit, optimize=opt, backend=backend)
            for (op, qubit) in zip(ops, qubits)
        ]

    return sum(ens).real


def _compute_approx_energy(
    theta,
    graph,
    qaoa_version,
    max_bond,
    opt=None,
    backend="numpy",
    mps=False,
):
    """
    UNTESTED. Find the compressed expectation value of the problem Hamiltonian with given QAOA angles.

    Parameters:
    -----------
    theta: np.ndarray
        QAOA angles of the form (gamma_1, ..., gamma_p, beta_1, ..., beta_p).
    graph: ProblemGraph
        Graph representing the instance of the problem.
    max_bond: int
        Maximum bond in the contraction of the QAOA circuit.
    qaoa_version: str
        Type of QAOA circuit to create.
    opt: str, optional
        Contraction path optimizer. Default is Quimb's default optimizer.
    backend: str, optional
        Backend to use for the simulation of the QAOA circuit. Default is "numpy".
    mps: bool, optional

    Returns:
    --------
    energy: float
        Compressed expectation value of the problem Hamiltonian with the given QAOA angles.
    """

    depth = len(theta) // 2
    gammas = theta[:depth]
    betas = theta[depth:]

    hamil = hamiltonian(graph)
    ops, qubits = hamil.operators()

    ansatz = instantiate_ansatz(graph, depth, gammas, betas, qaoa_version)
    if mps:
        ens = []

        for op, qubit in zip(ops, qubits):
            contracted_value = ansatz.local_expectation(
                op, qubit, max_bond, optimize=opt, backend=backend
            )

            ens.append(contracted_value)

    else:
        ens = []

        for op, qubit in zip(ops, qubits):
            tn = ansatz.local_expectation_tn(op, qubit, simplify_sequence="")
            tn.compress_simplify(inplace=True)
            tn.hyperinds_resolve(inplace=True)

            contracted_value = tn.contract_compressed(optimize=opt, max_bond=max_bond)

            ens.append(contracted_value)

    return sum(ens).real


def minimize_energy(
    theta_ini,
    graph,
    qaoa_version,
    optimizer="SLSQP",
    opt=None,
    backend="numpy",
    max_bond=None,
    mps=False,
    tau=None,
    **ansatz_opts,
):
    """
    Minimize the expectation value of the problem Hamiltonian. The actual computation is not rehearsed - the contraction widths and costs of each energy term are not pre-computed.

    Parameters:
    -----------
    theta_ini: np.ndarray
        Initial QAOA angles of the form (gamma_1, ..., gamma_p, beta_1, ..., beta_p).
    graph: ProblemGraph
        Graph representing the instance of the problem.
    qaoa_version: str
        Type of QAOA circuit to create.
    optimizer: str, optional
        SciPy optimizer to use for the minimization of the energy. Default is "SLSQP".
    opt: str, optional
        Contraction path optimizer. Default is Quimb's default optimizer.
    backend: str, optional
        Backend to use for the simulation of the QAOA circuit. Default is "numpy".
    max_bond: int, optional
        Maximum bond in the contraction of the QAOA circuit. If None, the bond is not limited.
    mps: bool, optional
        If True, initialize the QAOA circuit as a Matrix Product State (MPS) instead of as a general tensor networks.
    tau: float, optional
        Thresold for the optimization of the energy. If the energy is below this value, the optimization stops.

    Returns:
    --------
    theta: np.ndarray
        Optimal QAOA angles.
    """

    def wrapper_compute_energy(
        theta,
        graph,
        qaoa_version,
        opt=None,
        backend="numpy",
        mps=False,
        max_bond=None,
        kwargs=None,
    ):
        return compute_energy(
            theta,
            graph,
            qaoa_version,
            opt=opt,
            backend=backend,
            mps=mps,
            max_bond=max_bond,
            **kwargs,
        )

    depth = len(theta_ini) // 2

    # bound QAOA angles to their respective ranges
    eps = 1e-6
    bounds = [(-np.pi / 2 + eps, np.pi / 2 - eps)] * depth + [
        (-np.pi / 4 + eps, np.pi / 4 - eps)
    ] * depth

    # arguments to pass to the SciPy optimizer
    args = (graph, qaoa_version, opt, backend, mps, max_bond, ansatz_opts)

    if tau is None:
        res = minimize(
            wrapper_compute_energy,
            x0=theta_ini,
            method=optimizer,
            bounds=bounds,
            args=args,
        )
        theta = res.x

    else:
        f_wrapper = Objective_Function_Wrapper(wrapper_compute_energy, tau)

        try:
            res = minimize(
                f_wrapper,
                x0=theta_ini,
                method=optimizer,
                callback=f_wrapper.stop,
                bounds=bounds,
                args=args,
            )
            theta = res.x
        except Trigger:
            theta = f_wrapper.best_x
    # print(res)
    return theta


class Objective_Function_Wrapper:
    """
    Function wrapper stopping the minimisation of the objective function when the value of the said function passes a user-defined threshold.

    Parameters:
    -----------
    f: callable
        Objective function to minimize.
    tau: float
        Thresold for the optimization of the energy. If the energy is below this value, the optimization stops.

    Returns:
    --------
    fval: float
        Value of the objective function.
    """

    def __init__(self, f, tau):
        self.fun = f  # set the objective function
        self.best_x = None
        self.best_func = np.inf
        self.tau = tau  # set the user-desired threshold

    def __call__(self, xk, *args):
        fval = self.fun(xk, *args)
        if fval < self.best_func:
            self.best_func = fval
            self.best_x = xk

        return fval

    def stop(self, *args):
        if self.best_func <= self.tau:
            print("Optimization terminated: Desired approximation ratio achieved.")
            raise Trigger


class Trigger(Exception):
    pass
