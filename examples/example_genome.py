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

numqubit = 3
seed = 12345

p = 1
ini_method = "tqa"
qaoa_version = "regular"
problem = "genome"
mps = False
optimizer = "SLSQP"
tau = None
backend = "numpy"
shots = 10000

cotengra_kwargs = {
    # "minimize":'flops',
    "methods": ["greedy", "kahypar"],
    "reconf_opts": {},
    "optlib": "random",
    "max_repeats": 32,
    "parallel": True,
    "max_time": "rate:1e6",
}

G = nx.erdos_renyi_graph(numqubit, 0.6, seed=seed)
G.numnodes = G.order()
G.terms = {(i, j): 1 for (i, j) in G.edges}

nx.draw(G, with_labels=True)
plt.show()

# MAIN

draw_qaoa_circ(G, p, qaoa_version=qaoa_version, problem=problem)

opt = ctg.ReusableHyperOptimizer(**cotengra_kwargs)
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

start = time.time()
counts, energy, theta, compute_time = QAOA_Launcher(
    G,
    p,
    qaoa_version=qaoa_version,
    ini_method=ini_method,
    problem=problem,
    mps=mps,
    optimizer=optimizer,
    tau=tau,
    backend=backend,
    cotengra_kwargs=cotengra_kwargs,
).run_qaoa(shots)
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
