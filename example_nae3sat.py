import matplotlib.pyplot as plt
import networkx as nx
import time

from qaoa_quimb.launcher import *
from qaoa_quimb.utils import *
from nae3sat import *

# parameters
numqubit = 4
alpha = 1
p = 2

reg = 3
n = 20
seed = 666

qaoa_kwargs={"p":p,
             "ini_method":'tqa',
             "qaoa_version":'regular',
             "problem":'nae3sat',
             "mps":False,
             "optimizer":'SLSQP',
             "backend":'numpy',
             "shots":1024,}

contegra_kwargs={
                "methods":['greedy', 'kahypar'], 
                "reconf_opts":{},
                "max_repeats":100,
                "parallel":True,
                "max_time":"rate:1e6",}

numcau = alpha*numqubit
# seed = 123
# G = bicubic_graph(numqubit, numcau, 3, 3, seed)
G = nx.random_regular_graph(reg, n, seed = seed)
G.numnodes = G.order()

draw_qaoa_circ(G, p)

opt = ctg.ReusableHyperOptimizer(**contegra_kwargs)
width, cost = rehearse_qaoa_circ(G, p, qaoa_version="regular", problem="nae3sat", mps=False, opt=opt, backend="numpy")

print(width, cost)

start = time.time()
prob, counts, energy, theta, compute_time = QAOA_Launcher(G, **qaoa_kwargs, contegra_kwargs=contegra_kwargs).run_qaoa()
end = time.time()

print(compute_time)

print("Energy :", energy)

print("Time :", end-start)

