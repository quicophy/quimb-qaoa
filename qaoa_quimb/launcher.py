"""
Launcher for QAOA.
"""


import quimb.tensor as qtn
from collections import Counter
import time

from .initialization import ini
from .circuit import create_qaoa_circ
from .mps import create_qaoa_mps
from .contraction import compute_energy, minimize_energy
from .utils import rehearse_qaoa_circ
from .decomp import *
from .circuit import *


class QAOA_Launcher:
    """
    This class regroups the main methods of the regular QAOA algorithm applied to a particular problem. It instantiates a QAOA object for a specific graph with the necessary properties "numnodes" and "edges".
    """

    def __init__(
        self,
        G,
        p,
        qaoa_version="regular",
        ini_method="tqa",
        problem="nae3sat",
        mps=False,
        max_bond=None,
        optimizer="SLSQP",
        tau=None,
        backend="numpy",
        opt=None,
    ):
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
        self.max_bond = max_bond
        self.optimizer = optimizer
        self.tau = tau
        self.backend = backend
        self.opt = opt

    def run_qaoa(self):
        """
        Run the qaoa.
        """

        start_path = time.time()
        # rehearse_qaoa_circ(
        #     self.G,
        #     self.p,
        #     qaoa_version=self.qaoa_version,
        #     problem=self.problem,
        #     mps=self.mps,
        #     opt=self.opt,
        #     backend=self.backend,
        # )
        end_path = time.time()

        start_ini = time.time()
        theta_ini = ini(
            self.G,
            self.p,
            self.ini_method,
            qaoa_version=self.qaoa_version,
            problem=self.problem,
            mps=self.mps,
            max_bond=self.max_bond,
            opt=self.opt,
            backend=self.backend,
        )
        end_ini = time.time()

        start_minim = time.time()
        theta = minimize_energy(
            theta_ini,
            self.p,
            self.G,
            tau=self.tau,
            qaoa_version=self.qaoa_version,
            problem=self.problem,
            mps=self.mps,
            max_bond=self.max_bond,
            optimizer=self.optimizer,
            opt=self.opt,
            backend=self.backend,
        )
        end_minim = time.time()

        start_energy = time.time()
        energy = compute_energy(
            theta,
            self.p,
            self.G,
            qaoa_version=self.qaoa_version,
            problem=self.problem,
            mps=self.mps,
            max_bond=self.max_bond,
            opt=self.opt,
            backend=self.backend,
        )
        end_energy = time.time()

        compute_time = {
            "initialization": end_ini - start_ini,
            "contraction path": end_path - start_path,
            "minimisation": end_minim - start_minim,
            "energy": end_energy - start_energy,
        }

        return energy, theta, compute_time

    def run_and_sample_qaoa(self, shots, target_size=None):
        """
        Run and sample the qaoa.
        """

        energy, theta, compute_time = self.run_qaoa()

        if self.mps:
            psi0 = create_qaoa_mps(
                self.G,
                self.p,
                theta[: self.p],
                theta[self.p :],
                qaoa_version=self.qaoa_version,
                problem=self.problem,
            )
            circ = qtn.Circuit(self.G.numnodes, psi0=psi0)
        else:
            circ = create_qaoa_circ(
                self.G,
                self.p,
                theta[: self.p],
                theta[self.p :],
                qaoa_version=self.qaoa_version,
                problem=self.problem,
            )

        start_sampling = time.time()
        counts = Counter(circ.sample(shots, backend=self.backend, target_size=target_size))
        end_sampling = time.time()

        compute_time["sampling"] = end_sampling - start_sampling

        return counts, energy, theta, compute_time
