"""
Launcher for QAOA.
"""


import numpy as np
import cotengra as ctg
import quimb.tensor as qtn
import time

from .initialization import ini
from .circuit import create_qaoa_circ
from .mps import create_qaoa_mps
from .contraction import compute_energy, minimize_energy
from .gates import *
from .decomp import *


class QAOA_Launcher:
    """
    This class regroups the main methods of the regular QAOA algorithm applied to a particular problem. It instantiates a QAOA object for a specific graph with the necessary properties "numnodes" and "edges".
    """

    def __init__(self, G, p,
                qaoa_version='regular',
                ini_method="tqa",
                problem="nae3sat",
                mps=False,
                shots=1024,
                optimizer='SLSQP',
                backend="numpy",
                cotengra_kwargs={
                "methods":'kahypar', 
                "reconf_opts":{},
                "max_repeats":32,
                "parallel":True,
                "max_time":"rate:1e6",}):
        """
        Args:
        G: graph object
        theta_ini: initial list of unitary parameters
        A: parameter of the ising model
        shots: number of circuit samples
        optimizer: scipy optimizer
        simulator: qiskit optimizer
        """

        self.G = G
        self.p = p
        self.qaoa_version = qaoa_version
        self.ini_method = ini_method
        self.problem = problem
        self.mps = mps
        self.shots = shots
        self.optimizer = optimizer
        self.backend= backend
        self.opt = ctg.ReusableHyperOptimizer(**cotengra_kwargs)
        
    def run_qaoa(self, numcounts):
        """
        Run the qaoa.
        """

        start_ini = time.time()
        theta_ini = ini(self.G, self.p, self.ini_method,
                        qaoa_version=self.qaoa_version,
                        problem=self.problem,
                        mps=self.mps,
                        opt=self.opt)
        end_ini = time.time()

        start_minim = time.time()
        theta = minimize_energy(theta_ini, self.p, self.G,
                                qaoa_version=self.qaoa_version,
                                problem=self.problem,
                                mps=self.mps,
                                optimizer=self.optimizer,
                                opt=self.opt,
                                backend=self.backend)
        end_minim = time.time()

        if self.mps == False:
            circ = create_qaoa_circ(self.G, self.p,
                                    theta[:self.p], theta[self.p:],
                                    qaoa_version=self.qaoa_version,
                                    problem=self.problem)
        else:
            psi0 = create_qaoa_mps(self.G, self.p, 
                                theta[:self.p], theta[self.p:],
                                qaoa_version=self.qaoa_version,
                                problem=self.problem)
            circ = qtn.Circuit(self.G.numnodes, psi0=psi0)

        start_energy = time.time()
        energy = compute_energy(theta, self.p, self.G,
                                qaoa_version=self.qaoa_version,
                                problem=self.problem,
                                mps=self.mps,
                                opt=self.opt,
                                backend=self.backend)
        end_energy = time.time()

        start_sampling = time.time()
        counts = circ.sample(numcounts)
        end_sampling = time.time()

        compute_time = {
            "initialization": end_ini-start_ini,
            "minimisation": end_minim-start_minim,
            "energy": end_energy-start_energy,
            "sampling": end_sampling-start_sampling,
        }

        return counts, energy, theta, compute_time