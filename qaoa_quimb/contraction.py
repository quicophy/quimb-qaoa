"""
Functions for the contraction of QAOA.
"""


import numpy as np
from scipy.optimize import minimize

from .circuit import create_qaoa_circ
from .mps import create_qaoa_mps
from .hamiltonian import hamiltonian


def compute_energy(
    x,
    p,
    G,
    qaoa_version="regular",
    problem="nae3sat",
    mps=False,
    max_bond=None,
    opt=None,
    backend="numpy",
):
    """
    Computes the correct energy value of a qaoa circuit based on user input.
    """

    if max_bond is None:
        energy = compute_exact_energy(
            x,
            p,
            G,
            qaoa_version=qaoa_version,
            problem=problem,
            mps=mps,
            opt=opt,
            backend=backend,
        )
    else:
        energy = compute_approx_energy(
            x,
            p,
            G,
            max_bond,
            qaoa_version=qaoa_version,
            problem=problem,
            mps=mps,
            opt=opt,
            backend=backend,
        )
    return energy


def compute_exact_energy(
    x,
    p,
    G,
    qaoa_version="regular",
    problem="nae3sat",
    mps=False,
    opt=None,
    backend="numpy",
):
    """
    Find the expectation value of the problem Hamiltonian with the unitary parameters.

    Args:
        x: list of unitary parameters
    """

    gammas = x[:p]
    betas = x[p:]

    hamil = hamiltonian(G, problem)
    ops, qubits = hamil.operators()

    if mps:
        psi = create_qaoa_mps(G, p, gammas, betas, qaoa_version, problem=problem)
        ens = [
            psi.local_expectation_exact(op, qubit, optimize=opt, backend=backend)
            for (op, qubit) in zip(ops, qubits)
        ]

    else:
        circ = create_qaoa_circ(G, p, gammas, betas, qaoa_version, problem=problem)
        ens = [
            circ.local_expectation(op, qubit, optimize=opt, backend=backend)
            for (op, qubit) in zip(ops, qubits)
        ]

    return sum(ens).real


def compute_approx_energy(
    x,
    p,
    G,
    max_bond,
    qaoa_version="regular",
    problem="nae3sat",
    mps=False,
    opt=None,
    backend="numpy",
):
    """
    UNTESTED. Find the compressed expectation value of the problem Hamiltonian with the unitary parameters.

    Args:
        x: list of unitary parameters
    """

    gammas = x[:p]
    betas = x[p:]

    hamil = hamiltonian(G, problem)
    ops, qubits = hamil.operators()

    if mps:
        psi = create_qaoa_mps(G, p, gammas, betas, qaoa_version, problem=problem)

        ens = []

        for op, qubit in zip(ops, qubits):
            contracted_value = psi.local_expectation(
                op, qubit, max_bond, optimize=opt, backend=backend
            )

            ens.append(contracted_value)

    else:
        circ = create_qaoa_circ(G, p, gammas, betas, qaoa_version, problem=problem)

        ens = []

        for op, qubit in zip(ops, qubits):
            tn = circ.local_expectation_tn(op, qubit, simplify_sequence="")
            tn.compress_simplify(inplace=True)
            tn.hyperinds_resolve(inplace=True)

            contracted_value = tn.contract_compressed(optimize=opt, max_bond=max_bond)

            ens.append(contracted_value)

    return sum(ens).real


def minimize_energy(
    theta_ini,
    p,
    G,
    tau=None,
    qaoa_version="regular",
    problem="nae3sat",
    mps=False,
    max_bond=None,
    optimizer="SLSQP",
    opt=None,
    backend="numpy",
):
    """
    Minimize the expectation value of the problem Hamiltonian. The actual computation is not rehearsed - the contraction widths and costs of each energy term are not pre-computed.
    """

    eps = 1e-6
    # bounds = (
    #     [(-np.pi / 2 + eps, np.pi / 2 - eps)] * p +
    #     [(-np.pi / 4 + eps, np.pi / 4 - eps)] * p
    # )

    args = (p, G, qaoa_version, problem, mps, max_bond, opt, backend)

    if tau is None:
        res = minimize(compute_energy, x0=theta_ini, method=optimizer, args=args)
        theta = res.x

    else:
        f_wrapper = Objective_Function_Wrapper(compute_energy, tau)

        try:
            res = minimize(
                f_wrapper,
                x0=theta_ini,
                method=optimizer,
                callback=f_wrapper.stop,
                args=args,
            )
            theta = res.x
        except Trigger:
            theta = f_wrapper.best_x

    return theta


class Objective_Function_Wrapper:
    """
    Function wrapper stopping the minimisation of the objective function when the value of the said function passes a user-defined threshold.
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
