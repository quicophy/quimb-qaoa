"""
Functions for the contraction of QAOA.
"""


import numpy as np
import quimb as qu
from scipy.optimize import minimize
import cotengra as ctg

from .circuit import create_qaoa_circ
from .mps import create_qaoa_mps
from .hamiltonian import hamiltonian_gates, hamiltonian_ops


def compute_energy(x, p, G, qaoa_version="regular",
                            problem="nae3sat",
                            mps=False,
                            opt=None,
                            backend="numpy"):
    """
    Compute the energy of a qaoa circuit or a qaoa mps based on user input.
    """

    if mps == False:
        return compute_energy_circ(x, p, G, qaoa_version=qaoa_version,
                                            problem=problem,
                                            opt=opt,
                                            backend=backend)
    else:
        return compute_energy_mps(x, p, G, qaoa_version=qaoa_version,
                                            problem=problem,
                                            opt=opt,
                                            backend=backend)
    
def compute_energy_circ(x, p, G,
           qaoa_version="regular",
           problem="nae3sat",
           opt=None,
           backend="numpy"):
    """
    Find the expectation value of the problem Hamiltonian with the mps unitary parameters.

    Args:
        x: list of unitary parameters
    """

    gammas = x[:p]
    betas = x[p:]

    ops, qubits = hamiltonian_ops(G, problem=problem)

    circ = create_qaoa_circ(G, p, gammas, betas, qaoa_version, problem=problem)
    
    ens = [
        circ.local_expectation(op, qubit, optimize=opt, backend=backend)
            for (op, qubit) in zip(ops, qubits)
    ]

    return sum(ens).real

def compute_energy_mps(x, p, G,
           qaoa_version="regular",
           problem="nae3sat",
           opt=None,
           backend="numpy"):
    """
    Find the expectation value of the problem Hamiltonian with the circuit unitary parameters.

    Args:
        x: list of unitary parameters
    """

    gammas = x[:p]
    betas = x[p:]

    ops, qubits = hamiltonian_ops(G, problem=problem)

    psi = create_qaoa_mps(G, p, gammas, betas, qaoa_version, problem=problem)

    ens = [
        psi.local_expectation_exact(op, qubit, optimize=opt, backend=backend)
            for (op, qubit) in zip(ops, qubits)
    ]

    return sum(ens).real
    
def minimize_energy(theta_ini, p, G, 
                    tau=0.8,
                    qaoa_version="regular",
                    problem="nae3sat",
                    mps=False,
                    optimizer="SLSQP",
                    opt=None,
                    backend="numpy"):
        """
        Minimize the expectation value of the problem Hamiltonian. The actual computation is not rehearsed - the contraction widths and costs of each energy term are not pre-computed.
        """

        n = G.numnodes
        tau = -tau*n

        args = (p, G, qaoa_version, problem, mps, opt, backend)

        f_wrapper = Objective_Function_Wrapper(compute_energy, tau)

        try:
            res = minimize(f_wrapper,
                    x0 = theta_ini,
                    method = optimizer,
                    callback = f_wrapper.stop,
                    args=args)
            theta = res.x
        except Trigger:
            theta = f_wrapper.best_x

        return theta

class Objective_Function_Wrapper:
    """
    Function wrapper stopping the minimisation of the objective function when the value of the said function passes a user-defined threshold. 
    """

    def __init__(self, f, tau):
        self.fun = f                     # set the objective function
        self.best_x = None
        self.best_func = np.inf
        self.tau = tau                   # set the user-desired threshold

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