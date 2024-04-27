"""
Implementation of differents problems to be solved using QAOA.
"""

import igraph as ig
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import qecstruct as qs
from matplotlib.lines import Line2D


def problem_graph(problem, numvar, numcau, vardeg, caudeg, seed):
    """
    Instantiate a problem graph.

    Parameters
    ----------
    problem : str
        Name of the problem.
    numvar : int
        Number of variables.
    numcau : int
        Number of clauses.
    vardeg : int
        Variables degree.
    caudeg : int
        Clauses degree.
    seed : int
        Seed for random number generation.

    Returns
    -------
    problem_graph : ProblemGraph
        Problem graph.
    """

    if problem == "nae3sat":
        return MonoNaeThreeSatGraph(numvar, numcau, vardeg, caudeg, seed)
    elif problem == "mono1in3sat":
        return MonoOneInThreeSatGraph(numvar)
    elif problem == "mono2sat":
        return MonoTwoSatGraph(numvar)
    elif problem == "2sat":
        return TwoSatGraph(numvar)
    else:
        raise ValueError("The problem is not implemented.")


class ProblemGraph:
    """
    Base class for problem graphs.

    Attributes
    ----------
    problem : str
        Name of the problem.
    num_nodes : int
        Number of variables.
    cnf : numpy.ndarray
        CNF formula.
    edges : numpy.ndarray
        Edges of the Ising graph.
    terms : dict[Tuple[int, int], int]
        Couplings of the Ising graph, where the keys are the edges of the graph and the values are the weights.
    """

    def __init__(self):
        self.problem = None
        self.num_nodes = None
        self.cnf_ini = None
        self.cnf = None
        self.edges = None
        self.terms = None

    def cnf_view(self):
        """CNF formula view."""

        # Create adjacency matrix
        num_clauses = len(self.cnf_ini)
        num_variables = max(
            abs(literal) for clause in self.cnf_ini for literal in clause
        )
        adj_matrix = np.zeros((num_clauses, num_variables), dtype=int)

        for i, clause in enumerate(self.cnf_ini):
            adj_matrix[i, np.abs(clause) - 1] = 1

        # Prepare graph and add nodes
        graph = nx.Graph()
        graph.add_nodes_from(range(num_variables), bipartite=0, color="blue")
        graph.add_nodes_from(
            range(num_variables, num_variables + num_clauses), bipartite=1, color="red"
        )

        # Add edges based on the adjacency matrix
        edges = [
            (j, num_variables + i)
            for i in range(num_clauses)
            for j in range(num_variables)
            if adj_matrix[i, j]
        ]
        graph.add_edges_from(edges)

        # Graph layout and drawing
        pos = nx.spring_layout(graph)
        nx.draw(
            graph,
            pos,
            node_color=[data["color"] for _, data in graph.nodes(data=True)],
            with_labels=True,
            node_size=500,
        )

        # Legend setup
        plt.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label="Clauses",
                    markerfacecolor="red",
                    markersize=10,
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label="Variables",
                    markerfacecolor="blue",
                    markersize=10,
                ),
            ],
            loc="upper right",
        )

    def ising_view(self):
        """Ising formulation view."""

        # Create the graph from edges defined in the class
        graph = nx.Graph(list(self.edges))

        nx.draw(graph, with_labels=True, node_color="blue", node_size=500)


class MonoNaeThreeSatGraph(ProblemGraph):
    """
    This class instantiates a random bicubic graph representing a monotone NAE3SAT problem using qecstruct. It then maps the bicubic graph to an Ising graph using the Ising formulation of the NAE3SAT problem.

    Parameters
    ----------
    numvar : int
        Number of variables.
    numcau : int
        Number of clauses.
    vardeg : int
        Variables degree.
    caudeg : int
        Clauses degree.
    seed : int
        Seed for random number generation.

    Attributes
    ----------
    problem : str
        Name of the problem.
    num_nodes : int
        Number of variables.
    cnf_ini : numpy.ndarray
        3SAT formula.
    cnf : numpy.ndarray
        NAE3SAT formula.
    edges : numpy.ndarray
        Edges of the Ising graph.
    terms : dict[Tuple[int, int], int]
        Couplings of the Ising graph, where the keys are the edges of the graph and the values are the weights.
    """

    def __init__(self, numvar, numcau, vardeg, caudeg, seed):
        # samples a random bicubic graph
        code = qs.random_regular_code(numvar, numcau, vardeg, caudeg, qs.Rng(seed))

        # write the 3SAT formula and find the edges of the ising graph
        cnf_ini = []
        edges = []
        for row in code.par_mat().rows():
            temp_cnf = []
            for value in row:
                temp_cnf.append(value)
            cnf_ini.append(sorted(temp_cnf))
            edges.append([temp_cnf[0], temp_cnf[1]])
            edges.append([temp_cnf[1], temp_cnf[2]])
            edges.append([temp_cnf[2], temp_cnf[0]])

        # sort for consistency
        cnf_ini = sorted(cnf_ini)
        edges = sorted(edges)

        # name of the problem
        self.problem = "nae3sat"
        # number of variables
        self.num_nodes = numvar
        # 3SAT formula
        self.cnf_ini = np.array(cnf_ini) + 1
        # NAE3SAT formula
        self.cnf = np.vstack((self.cnf_ini, np.invert(self.cnf_ini) + 1))
        # edges of the ising graph
        self.edges = np.array(edges)
        # dictionary of edges of the ising graph
        terms = {}
        for i, j in self.edges:
            terms[(i, j)] = terms.get((i, j), 0) + 1
        self.terms = terms


class MonoOneInThreeSatGraph(ProblemGraph):
    """
    This class instantiates a random bicubic graph representating a monotone 1-in-3SAT problem. It then maps the bicubic graph to an Ising graph using the Ising formulation of the monotone 1-in-3SAT problem. ONLY SUPPORTS ALPHA = 2/3.

    Parameters
    ----------
    numvar : int
        Number of variables.

    Attributes
    ----------
    problem : str
        Name of the problem.
    num_nodes : int
        Number of variables.
    cnf_ini : numpy.ndarray
        3SAT formula.
    cnf : numpy.ndarray
        1in3SAT formula.
    edges : numpy.ndarray
        Edges of the Ising graph.
    terms : dict[Tuple[int, int], int]
        Couplings of the Ising graph, where the keys are the edges of the graph and the values are the weights.
    """

    def __init__(self, numvar):
        if numvar % 3 != 0:
            raise ValueError("The number of variable should be a multiple of 3.")

        numvar = int(2 * numvar / 3)

        cg = ig.Graph.Degree_Sequence([3] * numvar, method="vl")
        temp_edgelist = cg.get_edgelist()

        edgelist = []
        new_var = numvar
        for i, j in temp_edgelist:
            edgelist.append((new_var, i))
            edgelist.append((new_var, j))
            new_var += 1

        temp_cnf = []
        for var in range(numvar):
            temp = []
            for i, j in edgelist:
                if i == var:
                    temp.append(j)
                if j == var:
                    temp.append(i)
            temp_cnf.append(sorted(temp))

        cnf = np.array(sorted(temp_cnf)) - numvar

        # write the 3SAT formula and find the edges of the ising graph
        edges = []
        terms = {}
        for tpcnf in cnf:
            edges.append([tpcnf[0], tpcnf[1]])
            edges.append([tpcnf[1], tpcnf[2]])
            edges.append([tpcnf[2], tpcnf[0]])
            terms[(tpcnf[0], tpcnf[1])] = terms.get((tpcnf[0], tpcnf[1]), 0) + 1
            terms[(tpcnf[1], tpcnf[2])] = terms.get((tpcnf[1], tpcnf[2]), 0) + 1
            terms[(tpcnf[2], tpcnf[0])] = terms.get((tpcnf[2], tpcnf[0]), 0) + 1
            terms[(tpcnf[0],)] = terms.get((tpcnf[0],), 0) - 1
            terms[(tpcnf[1],)] = terms.get((tpcnf[1],), 0) - 1
            terms[(tpcnf[2],)] = terms.get((tpcnf[2],), 0) - 1

        # sort for consistency
        edges = sorted(edges)

        # name of the problem
        self.problem = "mono1in3sat"
        # 3SAT formula
        self.cnf_ini = cnf + 1
        # 1in3SAT formula
        cnf = []
        for i, j, k in self.cnf_ini.tolist():
            cnf.append((i, j, k))
            cnf.append((i, -j, -k))
            cnf.append((-i, -j, k))
            cnf.append((-i, j, -k))
            cnf.append((-i, -j, -k))
        self.cnf = np.array(cnf)
        # edges of the ising graph
        self.edges = np.array(edges)
        # number of variables
        self.num_nodes = int(3 * numvar / 2)
        # dictionary of edges of the ising graph
        self.terms = terms


class MonoTwoSatGraph(ProblemGraph):
    """
    This class instantiates a random bicubic graph representating a monotone 2-SAT problem. It then maps the bicubic graph to an Ising graph using the Ising formulation of the monotone 2-SAT problem. ONLY SUPPORTS ALPHA = 3/2.

    Parameters
    ----------
    numvar : int
        Number of variables.

    Attributes
    ----------
    problem : str
        Name of the problem.
    num_nodes : int
        Number of variables.
    cnf : numpy.ndarray
        2SAT formula.
    edges : numpy.ndarray
        Edges of the Ising graph.
    terms : dict[Tuple[int, int], int]
        Couplings of the Ising graph, where the keys are the edges of the graph and the values are the weights.
    """

    def __init__(self, numvar):
        cg = ig.Graph.Degree_Sequence([3] * numvar, method="vl")
        edgelist = sorted(cg.get_edgelist())
        cnf = np.array(edgelist) + 1

        terms = {}
        for i, j in edgelist:
            terms[(i,)] = terms.get((i,), 0) + 1
            terms[(j,)] = terms.get((j,), 0) + 1
            terms[(i, j)] = terms.get((i, j), 0) + 1

        # 2SAT formula
        self.cnf_ini = cnf
        self.cnf = cnf
        # edges of the ising graph
        self.edges = np.array(edgelist)
        # number of variables
        self.num_nodes = numvar
        # dictionary of edges of the ising graph
        self.terms = terms
        self.problem = "mono2sat"


class TwoSatGraph(ProblemGraph):
    """
    This class instantiates a random bicubic graph representating a 2-SAT problem. It then maps the bicubic graph to an Ising graph using the Ising formulation of the monotone 2-SAT problem. ONLY SUPPORTS ALPHA = 2/3.

    Parameters
    ----------
    numvar : int
        Number of variables.

    Attributes
    ----------
    problem : str
        Name of the problem.
    num_nodes : int
        Number of variables.
    cnf : numpy.ndarray
        2SAT formula.
    edges : numpy.ndarray
        Edges of the Ising graph.
    terms : dict[Tuple[int, int], int]
        Couplings of the Ising graph, where the keys are the edges of the graph and the values are the weights.
    """

    def __init__(self, numvar):
        cg = ig.Graph.Degree_Sequence([3] * numvar, method="vl")
        edgelist = sorted(cg.get_edgelist())
        cnf = np.array(edgelist) + 1

        # add negations
        negations = np.random.choice([-1, 1], size=cnf.shape)
        cnf = cnf * negations

        terms = {}
        for iter, (i, j) in enumerate(edgelist):
            terms[(i,)] = terms.get((i,), 0) + 1 * np.sign(cnf[iter, 0])
            terms[(j,)] = terms.get((j,), 0) + 1 * np.sign(cnf[iter, 1])
            terms[(i, j)] = terms.get((i, j), 0) + 1 * np.sign(
                cnf[iter, 0] * cnf[iter, 1]
            )

        # 3SAT formula
        self.cnf_ini = cnf
        self.cnf = cnf
        # edges of the ising graph
        self.edges = np.array(edgelist)
        # number of variables
        self.num_nodes = numvar
        # dictionary of edges of the ising graph
        self.terms = terms
        self.problem = "mono2sat"
