"""
Example for solving the NAE 3-SAT problem with QAOA.
"""


import numpy as np
import cotengra as ctg
import qecstruct as qs
import time

from qaoa_quimb.launcher import QAOA_Launcher
from qaoa_quimb.utils import draw_qaoa_circ, rehearse_qaoa_circ


class bicubic_graph:
    """
    This class instantiates a random bicubic graph representating a NAE3SAT problem using qecstruct. It then maps the bicubic graph to an Ising graph using the Ising formulation of the NA3SAT problem.
    """

    def __init__(self, numvar, numcau, vardeg, caudeg, seed):
        """
        Args:
        numvar: number of variables
        numcau: number of causes
        vardeg: variables degree
        caudeg: causes degree
        """

        # samples a random bicubic graph
        code = qs.random_regular_code(numvar, numcau, vardeg, caudeg, qs.Rng(seed))

        # write the 3SAT formula and find the edges of the ising graph
        cf = []
        edges = []
        for row in code.par_mat().rows():
            temp_cf = []
            for value in row:
                temp_cf.append(value)
            cf.append(temp_cf)
            edges.append([temp_cf[0], temp_cf[1]])
            edges.append([temp_cf[1], temp_cf[2]])
            edges.append([temp_cf[2], temp_cf[0]])
        edges = sorted(edges)

        # 3SAT formula
        self.cf = np.array(cf) + 1
        # NA3SAT formula
        self.cf_nae = np.vstack((self.cf, np.invert(self.cf) + 1))
        # edges of the ising graph
        self.edges = np.array(edges)
        # number of variables
        self.numnodes = numvar
        # dictionary of edges of the ising graph
        terms = {}
        for i, j in edges:
            terms[(i, j)] = terms.get((i, j), 0) + 1
        self.terms = terms


# PARAMETERS

# problem parameters
numqubit = 5
alpha = 1
p = 1
ini_method = "tqa"
qaoa_version = "regular"
problem = "nae3sat"
seed = 12345

# optimization parameters
mps = False
optimizer = "SLSQP"
backend = "numpy"
shots = 10000
tau = -0.8*numqubit*alpha

# slicing and compression parameters
target_size = None
max_bond = 8

# cotengra parameters
cotengra_kwargs = {
    "minimize":'flops',
    "methods": ["kahypar"],
    "reconf_opts": {},
    "optlib": "random",
    "max_repeats": 32,
    "parallel": True,
    "max_time": "rate:1e6",
}

opt = ctg.ReusableHyperOptimizer(**cotengra_kwargs)

if target_size is not None:
    cotengra_kwargs.pop("reconf_opts")
    cotengra_kwargs["slicing_reconf_opts"] = {"target_size":target_size,}

    opt = ctg.ReusableHyperOptimizer(**cotengra_kwargs)

if max_bond is not None:
    cotengra_kwargs.pop("reconf_opts")
    cotengra_kwargs["chi"] = max_bond
    cotengra_kwargs["minimize"] = 'max-compressed'
    cotengra_kwargs["methods"] = ['greedy-compressed', 'greedy-span']

    opt = ctg.ReusableHyperCompressedOptimizer(**cotengra_kwargs)

                                        
# REHEARSAL AND PREPARATION

numcau = alpha * numqubit
G = bicubic_graph(numqubit, numcau, 3, 3, seed)

draw_qaoa_circ(G, p, qaoa_version=qaoa_version, problem=problem)

width, cost = rehearse_qaoa_circ(
    G,
    p,
    qaoa_version=qaoa_version,
    problem=problem,
    mps=mps,
    opt=opt,
    backend=backend,
)

print("Width :", width)
print("Cost :", cost)


# MAIN

start = time.time()
counts, energy, theta, compute_time = QAOA_Launcher(
    G,
    p,
    qaoa_version=qaoa_version,
    ini_method=ini_method,
    problem=problem,
    mps=mps,
    max_bond=max_bond,
    optimizer=optimizer,
    tau=tau,
    backend=backend,
    opt=opt,
).run_and_sample_qaoa(shots, target_size=target_size)
end = time.time()

print("Energy :", energy)
print("Time :", end - start)
print("Total computation time :", compute_time)
