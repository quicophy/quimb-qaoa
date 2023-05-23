"""
Initialization methods for the initial parameters of QAOA.
"""


import numpy as np

from .contraction import compute_energy


def ini(G, p, ini_method,
        qaoa_version="regular",
        problem="nae3sat",
        mps=False,
        opt=None,
        backend="numpy"):
    """
    Creates the correct initial parameters based on user input.
    """

    if ini_method == 'random':
        theta_ini = rand_ini(p)
    elif ini_method == 'tqa':
        theta_ini = TQA_ini(G, p, 
                            qaoa_version=qaoa_version,
                            problem=problem,
                            mps=mps,
                            opt=opt,
                            backend=backend)
    else:
        raise ValueError('The initialization method is not valid.')

    return theta_ini

def rand_ini(p):
        """
        Creates a list of random initial unitary parameters for the QAOA algorithm.

        Args:
            p: depth of the QAOA circuit

        Returns:
            theta_ini: list of random unitary parameters
        """

        theta_ini = np.hstack((np.random.rand(p)*np.pi*2,np.random.rand(p)*np.pi*2))

        return theta_ini

def TQA_ini(G, p, 
            qaoa_version="regular",
            problem="nae3sat",
            mps=False,
            opt=None,
            backend="numpy"):
        """
        Creates a list of initial unitary parameters for the QAOA algorithm. The parameters are initialized based on the Trotterized Quantum Annealing (TQA) strategy for initialization. See "Quantum Annealing Initialization of the Quantum Approximate Optimization Algorithm".

        Returns:
            theta_ini: list of random unitary parameters
        """

        time = np.linspace(0.1, 3, 20)

        energies = []
        for t_max in time:
            dt = t_max/p
            t = dt*(np.arange(1, p+1)-0.5)
            gamma = (t/t_max)*dt
            beta = -(1-t/t_max)*dt
            
            theta = np.concatenate((gamma,beta))

            energies.append(compute_energy(theta, p, G, qaoa_version=qaoa_version, problem=problem ,mps=mps, opt=opt, backend=backend))

        idx = np.argmin(energies)
        t_max = time[idx]

        dt = t_max/p
        t = dt * (np.arange(1, p+1)-0.5)

        gamma = (t/t_max)*dt
        beta = -(1-t/t_max)*dt
        theta_ini = np.concatenate((gamma,beta))

        return theta_ini