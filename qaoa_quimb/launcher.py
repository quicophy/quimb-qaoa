#################################################################
#                                                               #
#   QUIMB IMPLEMENTATION OF QAOA                                #
#   ==========================================                  #
#   ( qaoa_quimb.py )                                           #
#   First instance: 05/01/2023                                  #
#   Written by Julien Drapeau (julien.drapeau@usherbrooke.ca)   #
#                                                               #
#   Implementation of the Quantum Approximate Optimization      #
#   Algorithm (QAOA) in Quimb. The following QAOA extensions    # 
#   are also implemented: Trotterized Quantum Annealing (TQA)   #
#   initialization, Grover-Mixer QAOA GM-QAOA).                 #
#                                                               #
#   DEPENDENCIES: numpy, quimb, cotengra, scipy                 #
#                                                               #
#################################################################


import numpy as np
import quimb.tensor as qtn
import time

from .initialization import *
from .hamiltonian import *
from .circuit import *
from .mps import *
from .contraction import *
from .gates import *
from .decomp import *
from .utils import *

class QAOA_Launcher:
    """
    This class regroups the main methods of the regular QAOA algorithm applied to a NAE3SAT problem. It instantiates a QAOA object for a specific graph representing a NAE3SAT problem.
    """

    def __init__(self, G, p,
                qaoa_version='regular',
                ini_method="tqa",
                problem="nae3sat",
                mps=False,
                shots=1024,
                optimizer='SLSQP',
                backend="numpy",
                contegra_kwargs={
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
        self.opt = ctg.ReusableHyperOptimizer(**contegra_kwargs)
        
    def run_qaoa(self):
        """
        Minimize the expectation value by finding the best parameters.
        Analyse the results with a histogram.

        Args:
            G: graph object
            p: int
            number of alternating unitairies
            A: parameter of the ising model

        Returns:
            obj: dict
                counts

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

        print("Done!")
        
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

        pre_counts = circ.simulate_counts(1000)
        # pre_prob = find_prob_sol(self.G, pre_counts)[0]
        pre_prob = 0.01

        start_sampling = time.time()
        # counts = circ.simulate_counts(int(10*1/pre_prob*self.G.numnodes**4))
        counts = 1
        end_sampling = time.time()

        start_pdf = time.time()
        # prob = np.squeeze(np.abs(circ.to_dense())**2)
        prob = 1
        end_pdf = time.time()

        compute_time = {
            "initialization": end_ini-start_ini,
            "minimisation": end_minim-start_minim,
            "energy": end_energy-start_energy,
            "sampling": end_sampling-start_sampling,
            "probability-density-function": end_pdf-start_pdf
        }

        return prob, counts, energy, theta, compute_time
    
