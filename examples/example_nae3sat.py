"""
Example for solving the NAE 3-SAT problem with QAOA.
"""


import cotengra as ctg
import matplotlib.pyplot as plt
import time

from quimb_qaoa.launcher import QAOALauncher
from quimb_qaoa.problem import Nae3satGraph
from quimb_qaoa.utils import draw_qaoa_circ, rehearse_qaoa_circ


# PARAMETERS

# problem parameters
numqubit = 6
alpha = 1
p = 3
ini_method = "tqa"
qaoa_version = "regular"
problem = "nae3sat"
seed = 666

# optimization parameters
contract_mps = True
sampling_mps = True
optimizer = "SLSQP"
backend = "numpy"
shots = 1000
tau = -0.9 * numqubit * alpha
tau = None

# slicing and compression parameters
target_size = None
max_bond = None


# CONTRACTION PATH PARAMETERS (CONTENGRA)

# contraction parameters
contract_kwargs = {
    "minimize": "flops",
    "methods": ["greedy"],
    "reconf_opts": {},
    "optlib": "random",
    "max_repeats": 32,
    "parallel": True,
    "max_time": "rate:1e6",
}

contract_opt = ctg.ReusableHyperOptimizer(**contract_kwargs)

if target_size is not None:
    contract_kwargs.pop("reconf_opts")
    contract_kwargs["slicing_reconf_opts"] = {
        "target_size": target_size,
    }

    contract_opt = ctg.ReusableHyperOptimizer(**contract_kwargs)

if max_bond is not None:
    contract_kwargs["chi"] = max_bond
    contract_kwargs["minimize"] = "max-compressed"
    contract_kwargs["methods"] = ["greedy-compressed", "greedy-span"]

    contract_opt = ctg.ReusableHyperCompressedOptimizer(**contract_kwargs)

# sampling parameters
sampling_kwargs = {
    "minimize": "flops",
    "methods": ["greedy"],
    "reconf_opts": {},
    "optlib": "random",
    "max_repeats": 32,
    "parallel": True,
    "max_time": "rate:1e6",
}

sampling_opt = ctg.ReusableHyperOptimizer(**sampling_kwargs)

if target_size is not None:
    sampling_kwargs.pop("reconf_opts")
    sampling_kwargs["slicing_reconf_opts"] = {
        "target_size": target_size,
    }

    sampling_opt = ctg.ReusableHyperOptimizer(**sampling_kwargs)

if max_bond is not None:
    sampling_kwargs["chi"] = max_bond
    sampling_kwargs["minimize"] = "max-compressed"
    sampling_kwargs["methods"] = ["greedy-compressed", "greedy-span"]

    sampling_opt = ctg.ReusableHyperCompressedOptimizer(**sampling_kwargs)


# REHEARSAL AND PREPARATION

numcau = alpha * numqubit
G = Nae3satGraph(numqubit, numcau, 3, 3, seed)
print("3-SAT formula:\n", G.cf_ini)

G.cnf_view()
plt.show()

G.ising_view()
plt.show()

draw_qaoa_circ(G, p, qaoa_version=qaoa_version)

width, cost, local_exp_rehs = rehearse_qaoa_circ(
    G,
    p,
    qaoa_version=qaoa_version,
    mps=contract_mps,
    opt=contract_opt,
    backend=backend,
    draw=True,
)

print("Width :", width)
print("Cost :", cost)


# MAIN

start = time.time()
QAOA = QAOALauncher(
    G,
    p,
    qaoa_version=qaoa_version,
    problem=problem,
    max_bond=max_bond,
    optimizer=optimizer,
    tau=tau,
    backend=backend,
)
theta_ini = QAOA.initialize_qaoa(
    ini_method=ini_method, opt=contract_opt, mps=contract_mps
)
print("Initialization is done!")
energy, theta = QAOA.run_qaoa(opt=contract_opt, mps=contract_mps)
print("Optimization is done!")
counts = QAOA.sample_qaoa(shots, opt=sampling_opt, mps=sampling_mps)
compute_time = QAOA.compute_time
print("Sampling is done!")
end = time.time()

print("Energy :", energy)
print("Time :", end - start)
print("Total computation time :", compute_time)
