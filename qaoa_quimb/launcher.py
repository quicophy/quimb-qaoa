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


class QAOALauncher:
    """
    This class regroups the main methods of the regular QAOA algorithm applied to a particular problem. It instantiates a QAOA object for a specific graph with the necessary properties "numnodes" and "edges".
    """

    def __init__(
        self,
        G,
        p,
        qaoa_version="regular",
        problem="nae3sat",
        max_bond=None,
        optimizer="SLSQP",
        tau=None,
        backend="numpy",
    ):
        """
        Args:
        G: graph object
        theta_ini: initial list of unitary parameters
        shots: number of circuit samples
        optimizer: scipy optimizer
        """

        self.G = G
        self.p = p
        self.qaoa_version = qaoa_version
        self.problem = problem
        self.max_bond = max_bond
        self.optimizer = optimizer
        self.tau = tau
        self.backend = backend
        self.compute_time = {}
        self.theta_ini = None
        self.theta_opt = None

    def initialize_qaoa(self, ini_method="tqa", opt=None, mps=False):
        """
        Initialize QAOA.
        """

        # start_path = time.time()
        # rehearse_qaoa_circ(
        #     self.G,
        #     self.p,
        #     qaoa_version=self.qaoa_version,
        #     problem=self.problem,
        #     mps=mps,
        #     opt=opt,
        #     backend=self.backend,
        # )
        # end_path = time.time()

        start_ini = time.time()
        theta_ini = ini(
            self.G,
            self.p,
            ini_method,
            qaoa_version=self.qaoa_version,
            problem=self.problem,
            mps=mps,
            max_bond=self.max_bond,
            opt=opt,
            backend=self.backend,
        )
        end_ini = time.time()

        # self.compute_time["contraction path"] = end_path - start_path
        self.compute_time["initialization"] = end_ini - start_ini

        self.theta_ini = theta_ini

        return theta_ini

    def run_qaoa(self, opt=None, mps=False):
        """
        Run the qaoa.
        """

        if self.theta_ini is None:
            raise ValueError("Please initialize QAOA before running.")

        start_minim = time.time()
        theta_opt = minimize_energy(
            self.theta_ini,
            self.p,
            self.G,
            tau=self.tau,
            qaoa_version=self.qaoa_version,
            problem=self.problem,
            mps=mps,
            max_bond=self.max_bond,
            optimizer=self.optimizer,
            opt=opt,
            backend=self.backend,
        )
        end_minim = time.time()

        start_energy = time.time()
        energy = compute_energy(
            theta_opt,
            self.p,
            self.G,
            qaoa_version=self.qaoa_version,
            problem=self.problem,
            mps=mps,
            max_bond=self.max_bond,
            opt=opt,
            backend=self.backend,
        )
        end_energy = time.time()

        self.compute_time["minimisation"] = end_minim - start_minim
        self.compute_time["energy"] = end_energy - start_energy

        self.energy = energy
        self.theta_opt = theta_opt

        return energy, theta_opt

    def sample_qaoa(self, shots, opt=None, mps=True):
        """
        Sample the qaoa.
        """

        if self.theta_opt is not None:
            theta = self.theta_opt
        elif self.theta_ini is not None:
            theta = self.theta_ini
        else:
            raise ValueError(
                "Please initialize or initialize and run QAOA before sampling."
            )

        if mps:
            psi0 = create_qaoa_mps(
                self.G,
                self.p,
                theta[: self.p],
                theta[self.p :],
                qaoa_version=self.qaoa_version,
                problem=self.problem,
            )
            circ = qtn.Circuit(psi0=psi0)
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
        counts = Counter(circ.sample(shots, optimize=opt, backend=self.backend))
        end_sampling = time.time()

        self.compute_time["sampling"] = end_sampling - start_sampling
        self.counts = counts

        return counts
