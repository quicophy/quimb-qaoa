"""
Example for solving the NAE 3-SAT problem with QAOA.
"""


import numpy as np
import networkx as nx
import cotengra as ctg
import qecstruct as qs
import matplotlib.pyplot as plt
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

        #samples a random bicubic graph
        code = qs.random_regular_code(numvar, numcau, vardeg, caudeg, qs.Rng(seed))

        #write the 3SAT formula and find the edges of the ising graph
        cf = []
        edges = []
        for row in code.par_mat().rows():
            temp_cf = []
            for value in row:
                temp_cf.append(value)
            cf.append(temp_cf)
            edges.append([temp_cf[0],temp_cf[1]])
            edges.append([temp_cf[1],temp_cf[2]])
            edges.append([temp_cf[2],temp_cf[0]])

        #3SAT formula
        self.cf = np.array(cf)+1
        #NA3SAT formula
        self.cf_nae = np.vstack((self.cf, np.invert(self.cf)+1))
        #edges of the ising graph
        self.edges = np.array(edges)
        #number of variables
        self.numnodes = numvar


# PARAMETERS

numqubit = 4
alpha = 1
seed = 123

p = 1
ini_method = 'tqa'
qaoa_version = 'regular'
problem = 'nae3sat'
mps = False
optimizer = 'SLSQP'
backend = 'numpy'
shots = 1024

cotengra_kwargs={
                # "minimize":'flops',
                "methods":['greedy', 'kahypar'], 
                "reconf_opts":{},
                "optlib":"random",
                "max_repeats":32,
                "parallel":True,
                "max_time":"rate:1e6",}

numcau = alpha*numqubit
G = bicubic_graph(numqubit, numcau, 3, 3, seed)

# MAIN

draw_qaoa_circ(G, p)

opt = ctg.ReusableHyperOptimizer(**cotengra_kwargs)
width, cost = rehearse_qaoa_circ(G, p, ini_method,
                                 qaoa_version=qaoa_version,
                                 problem=problem,
                                 mps=mps,
                                 opt=opt,
                                 backend=backend)

print("Width :", width)
print("Cost :", cost)

start = time.time()
counts, energy, theta, compute_time = QAOA_Launcher(G, p, 
                                        qaoa_version='regular',
                                        ini_method="tqa",
                                        problem="nae3sat",
                                        mps=False,
                                        shots=1024,
                                        optimizer='SLSQP',
                                        backend="numpy",
                                        cotengra_kwargs=cotengra_kwargs).run_qaoa(1000)
end = time.time()

print("Energy :", energy)
print("Time :", end-start)
print("Total computation time :", compute_time)