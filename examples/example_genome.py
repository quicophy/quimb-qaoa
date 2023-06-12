"""
Example for solving the genome assembly/travelling salesman problem with QAOA.
"""


import networkx as nx
import cotengra as ctg
import matplotlib.pyplot as plt
import time

from qaoa_quimb.launcher import QAOA_Launcher
from qaoa_quimb.utils import draw_qaoa_circ, rehearse_qaoa_circ


# PARAMETERS

# problem parameters
numqubit = 3
p = 1
ini_method = "tqa"
qaoa_version = "regular"
problem = "genome"
seed = 12345

# optimization parameters
mps = False
optimizer = "SLSQP"
backend = "numpy"
shots = 10000
tau = None

# slicing and compression parameters
target_size = None
max_bond = None

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
    cotengra_kwargs = {
        "chi": max_bond,
        "minimize": 'max_compressed',
        "methods": ['greedy-compressed', 'greedy-span'],
        "max_repeats": 32,
        "parallel": True,
        "max_time": "rate:1e6",
    }

    opt = ctg.ReusableHyperCompressedOptimizer(**cotengra_kwargs)


# REHEARSAL AND PREPARATION

G = nx.erdos_renyi_graph(numqubit, 0.6, seed=seed)
G.numnodes = G.order()
G.terms = {(i, j): 1 for (i, j) in G.edges}

nx.draw(G, with_labels=True)
plt.show()

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
