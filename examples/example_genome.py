"""
Example for solving the genome assembly/travelling salesman problem with QAOA.
"""


import networkx as nx
import cotengra as ctg
import matplotlib.pyplot as plt
import time

from qaoa_quimb.launcher import QAOALauncher
from qaoa_quimb.utils import draw_qaoa_circ, rehearse_qaoa_circ


# GENERAL PARAMETERS

# problem parameters
numqubit = 3
p = 1
ini_method = "tqa"
qaoa_version = "regular"
problem = "genome"
seed = 12345

# optimization parameters
contract_mps = False
sampling_mps = True
optimizer = "SLSQP"
backend = "numpy"
shots = 100
tau = None

# slicing and compression parameters
target_size = None
max_bond = None


# COTENGRA PARAMETERSz

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

G = nx.erdos_renyi_graph(numqubit, 0.6, seed=seed)
G.numnodes = G.order()
G.terms = {(i, j): 1 for (i, j) in G.edges}

nx.draw(G, with_labels=True)
plt.show()

draw_qaoa_circ(G, p, qaoa_version=qaoa_version, problem=problem)

width, cost, local_exp_rehs = rehearse_qaoa_circ(
    G,
    p,
    qaoa_version=qaoa_version,
    problem=problem,
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
print("Sampling is done!")
compute_time = QAOA.compute_time
end = time.time()

max_count = max(counts, key=counts.get)
print("Max count:", max_count)
print("Number of max count:", counts[max_count])

counts.pop(max_count)
max_count = max(counts, key=counts.get)
print("Max count:", max_count)
print("Number of max count:", counts[max_count])

counts.pop(max_count)
max_count = max(counts, key=counts.get)
print("Max count:", max_count)
print("Number of max count:", counts[max_count])

print("Energy :", energy)
print("Time :", end - start)
print("Total computation time :", compute_time)
