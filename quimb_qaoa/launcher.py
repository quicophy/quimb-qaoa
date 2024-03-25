"""
Launcher for QAOA. Main class for the simulation of QAOA circuits.
"""

import time
from collections import Counter

import quimb.tensor as qtn

from .circuit import create_qaoa_circ
from .contraction import compute_energy, minimize_energy
from .decomp import *
from .initialization import ini
from .mps import create_qaoa_mps
from .utils import rehearse_qaoa_circ


class QAOALauncher:
    """
    This class regroups the main methods of QAOA applied to a particular optimization problem. The launcher can initialize, run, and sample the QAOA circuit.

    Parameters:
    -----------
    graph: ProblemGraph
        Graph representing the instance of the problem from the following:
        - "maxcut": Max-Cut problem.
        - "nae3sat": Not-All-Equal 3-SAT problem.
        - "1in3sat": 1-in-3-SAT problem.
    depth: int
        Number of layers of gates to apply (depth 'p').
    qaoa_version: str
        Type of QAOA circuit to create from the following:
        - "regular": X-mixer QAOA circuit.
        - "grover-mixer": Grover-mixer QAOA circuit.
    optimizer: str, optional
        SciPy optimizer to use for the minimization of the energy. Default is "SLSQP".
    backend: str, optional
        Backend to use for the simulation of the QAOA circuit from Quimb's possible backends. Default is "numpy".
    max_bond: int, optional
        Maximum bond in the contraction of the QAOA circuit. If None, the bond is not limited.
    tau: float, optional
        Thresold for the optimization of the energy. If the energy is below this value, the optimization stops.

    Attributes:
    -----------
    theta_ini: np.ndarray
        Initial parameters of the QAOA circuit.
    theta_opt: np.ndarray
        Optimal parameters of the QAOA circuit.
    ansatz: qtn.Circuit
        QAOA ansatz instantiated (either in the circuit or MPS format).
    compute_time: dict
        Dictionary of the computation time of the different steps of the QAOA circuit.
    """

    def __init__(
        self,
        graph,
        depth,
        qaoa_version,
        optimizer="SLSQP",
        backend="numpy",
        max_bond=None,
        tau=None,
    ):
        self.graph = graph
        self.depth = depth
        self.qaoa_version = qaoa_version
        self.optimizer = optimizer
        self.backend = backend
        self.max_bond = max_bond
        self.tau = tau

        # QAOA angles of the form (gamma_1, ..., gamma_p, beta_1, ..., beta_p)
        self.theta_ini = None
        self.theta_opt = None

        self.compute_time = {
            "initialization": 0,
            "contraction path": 0,
            "energy": 0,
            "minimisation": 0,
            "sampling": 0,
        }

    def instantiate_qaoa(self, theta, mps=False, **ansatz_opts):
        """
        Instantiate the QAOA ansatz (circuit or MPS).

        Parameters:
        -----------
        theta: np.ndarray
            Parameters of the QAOA circuit to instantiate.
        mps: bool, optional
            If True, initialize the QAOA circuit as a Matrix Product State (MPS) instead of as a general tensor networks.

        Returns:
        --------
        circ: qtn.Circuit
            QAOA circuit instantiated.
        """

        # create the QAOA circuit or MPS
        if mps:
            psi0 = create_qaoa_mps(
                self.graph,
                self.depth,
                theta[: self.depth],
                theta[self.depth :],
                qaoa_version=self.qaoa_version,
                **ansatz_opts,
            )
            circ = qtn.Circuit(psi0=psi0)
        else:
            circ = create_qaoa_circ(
                self.graph,
                self.depth,
                theta[: self.depth],
                theta[self.depth :],
                qaoa_version=self.qaoa_version,
                **ansatz_opts,
            )

        return circ

    def initialize_qaoa(self, ini_method, opt=None, mps=False, **ansatz_opts):
        """
        Initialize QAOA.

        Parameters:
        -----------
        ini_method: str
            Method to use for the initialization of the QAOA circuit from the following:
            - "random": Random initialization of the parameters.
            - "tqa": Trotterized Quantum Annealing initialization of the parameters.
        opt: str, optional
            Contraction path optimizer. Default is Quimb's default optimizer.
        mps: bool, optional
            If True, initialize the QAOA circuit as a Matrix Product State (MPS) instead of as a general tensor networks.

        Returns:
        --------
        theta_ini: np.ndarray
            Initial parameters of the QAOA circuit.
        width: float
            Maximum width of the contraction of the QAOA circuit.
        cost: float
            Cost of the contraction of the QAOA circuit.
        """

        # rehearse the QAOA circuit to fin the contraction path
        start_path = time.time()
        width, cost = 0, 0
        width, cost, local_exp_rehs = rehearse_qaoa_circ(
            self.graph,
            self.depth,
            qaoa_version=self.qaoa_version,
            opt=opt,
            backend=self.backend,
            mps=mps,
            **ansatz_opts,
        )
        end_path = time.time()

        # initialize the QAOA circuit
        start_ini = time.time()
        theta_ini = ini(
            self.graph,
            self.depth,
            ini_method,
            qaoa_version=self.qaoa_version,
            opt=opt,
            backend=self.backend,
            mps=mps,
            max_bond=self.max_bond,
            **ansatz_opts,
        )
        end_ini = time.time()

        self.compute_time["contraction path"] = end_path - start_path
        self.compute_time["initialization"] = end_ini - start_ini

        # save the initial parameters
        self.theta_ini = theta_ini

        return theta_ini, width, cost

    def optimize_qaoa(self, opt=None, mps=False, energy=True, **ansatz_opts):
        """
        Optimize the qaoa.

        Parameters:
        -----------
        opt: str, optional
            Contraction path optimizer. Default is Quimb's default optimizer.
        mps: bool, optional
            If True, initialize the QAOA circuit as a Matrix Product State (MPS) instead of as a general tensor networks.

        Returns:
        --------
        energy: float
            Energy of the QAOA circuit.
        theta_opt: np.ndarray
            Optimal parameters of the QAOA circuit.
        """

        if self.theta_ini is None:
            raise ValueError("Please initialize QAOA before running.")

        # minimize the energy
        start_minim = time.time()
        theta_opt = minimize_energy(
            self.theta_ini,
            self.graph,
            qaoa_version=self.qaoa_version,
            optimizer=self.optimizer,
            opt=opt,
            backend=self.backend,
            max_bond=self.max_bond,
            mps=mps,
            tau=self.tau,
            **ansatz_opts,
        )
        end_minim = time.time()

        if energy:
            # compute the final energy (useful for contraction time)
            start_energy = time.time()
            energy = compute_energy(
                theta_opt,
                self.graph,
                qaoa_version=self.qaoa_version,
                opt=opt,
                backend=self.backend,
                mps=mps,
                max_bond=self.max_bond,
                **ansatz_opts,
            )
            end_energy = time.time()
        else:
            start_energy = None
            end_energy = None
            energy = None

        self.compute_time["minimisation"] = end_minim - start_minim
        self.compute_time["energy"] = end_energy - start_energy

        # save the optimal parameters
        self.energy = energy
        self.theta_opt = theta_opt

        return energy, theta_opt

    def sample_qaoa(self, shots, ansatz=None, opt=None, mps=True, **ansatz_opts):
        """
        Sample the qaoa.

        Parameters:
        -----------
        shots: int
            Number of samples to take.
        ansatz: qtn.Circuit, optional
            QAOA ansatz to sample. Used to keep the marginals found. If None, the QAOA ansatz is instantiated from the optimal parameters.
        opt: str, optional
            Contraction path optimizer. Default is Quimb's default optimizer.
        mps: bool, optional
            If True, initialize the QAOA circuit as a Matrix Product State (MPS) instead of as a general tensor networks.

        Returns:
        --------
        counts: Counter
            Counter of the samples.
        """

        if ansatz is None:
            if self.theta_opt is not None:
                ansatz = self.instantiate_qaoa(self.theta_opt, mps, **ansatz_opts)
            elif self.theta_ini is not None:
                ansatz = self.instantiate_qaoa(self.theta_ini, mps, **ansatz_opts)
            else:
                raise ValueError(
                    "Please initialize or initialize and run QAOA before sampling."
                )

        # sample the QAOA circuit
        start_sampling = time.time()
        # TO CHANGE
        # counts = Counter(ansatz.sample(shots, optimize=opt, backend=self.backend, max_marginal_storage=2**28))
        counts = Counter(
            ansatz.simulate_counts(shots, optimize=opt, backend=self.backend)
        )
        end_sampling = time.time()

        self.compute_time["sampling"] = end_sampling - start_sampling
        self.counts = counts

        return counts
