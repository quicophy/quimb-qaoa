"""
Implementation of differents problems to be solved using QAOA.
"""


import numpy as np
import qecstruct as qs
import networkx as nx
import igraph as ig


class Nae3satGraph:
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
    numnodes : int
        Number of variables.
    cf_ini : numpy.ndarray
        3SAT formula.
    cf : numpy.ndarray
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
        cf_ini = []
        edges = []
        for row in code.par_mat().rows():
            temp_cf = []
            for value in row:
                temp_cf.append(value)
            cf_ini.append(temp_cf)
            edges.append([temp_cf[0], temp_cf[1]])
            edges.append([temp_cf[1], temp_cf[2]])
            edges.append([temp_cf[2], temp_cf[0]])

        # name of the problem
        self.problem = "nae3sat"
        # number of variables
        self.numnodes = numvar
        # 3SAT formula
        self.cf_ini = np.array(cf_ini) + 1
        # NAE3SAT formula
        self.cf = np.vstack((self.cf_ini, np.invert(self.cf_ini) + 1))
        # edges of the ising graph
        self.edges = np.array(edges)
        # dictionary of edges of the ising graph
        terms = {}
        for i, j in self.edges:
            terms[(i, j)] = terms.get((i, j), 0) + 1
        self.terms = terms

    def cnf_view(self):
        """CNF formula view."""

        numcau = len(self.cf_ini)
        numvar = np.array([abs(np.array(list(i))).max() for i in self.cf_ini]).max()

        adj_mat = np.zeros([numcau, numvar], int)
        for i, r in enumerate(self.cf_ini):
            adj_mat[i][abs(np.array(r)) - 1] = 1

        graph = nx.Graph()

        # add nodes with the bipartite attribute
        graph.add_nodes_from(
            range(numvar),
            bipartite=0,
            color="blue",
            label={i: f"{i+1}" for i in range(numvar)},
        )  # variable nodes
        graph.add_nodes_from(
            range(numvar, numvar + numcau),
            bipartite=1,
            color="red",
            label={i + numvar: f"{i+1}" for i in range(numcau)},
        )  # clause nodes

        # Add edges based on the biadjacency matrix
        for i in range(numcau):
            for j in range(numvar):
                if adj_mat[i][j] == 1:
                    graph.add_edge(j, numvar + i)  # Connect clauses to variables

        pos = nx.spring_layout(graph)  # Generate position for each node
        colors = [
            graph.nodes[n]["color"] for n in graph.nodes
        ]  # Color based on the node attribute

        # Draw nodes and edges
        nx.draw(graph, pos, node_color=colors, with_labels=False, node_size=700)

        # Draw custom labels
        custom_labels = {}
        for node in graph.nodes(data=True):
            custom_labels[node[0]] = (
                node[1]["label"][node[0]] if "label" in node[1] else node[0]
            )

        nx.draw_networkx_labels(graph, pos, labels=custom_labels)

    def ising_view(self):
        """Ising formulation graph."""

        nx.Graph(list(self.edges))


class Mono1in3satGraph:
    """
    This class instantiates a random bicubic graph representating a monotone 1-in-3SAT problem using qecstruct. It then maps the bicubic graph to an Ising graph using the Ising formulation of the monotone 1-in-3SAT problem.

    Parameters
    ----------
    numvar : int
        Number of variables.

    Attributes
    ----------
    problem : str
        Name of the problem.
    numnodes : int
        Number of variables.
    cf_ini : numpy.ndarray
        3SAT formula.
    cf : numpy.ndarray
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

        temp_cf = []
        for var in range(numvar):
            temp = []
            for i, j in edgelist:
                if i == var:
                    temp.append(j)
                if j == var:
                    temp.append(i)
            temp_cf.append(temp)

        cf = np.array(temp_cf) - numvar

        # write the 3SAT formula and find the edges of the ising graph
        edges = []
        terms = {}
        for tpcf in cf:
            edges.append([tpcf[0], tpcf[1]])
            edges.append([tpcf[1], tpcf[2]])
            edges.append([tpcf[2], tpcf[0]])
            terms[(tpcf[0], tpcf[1])] = terms.get((tpcf[0], tpcf[1]), 0) + 1
            terms[(tpcf[1], tpcf[2])] = terms.get((tpcf[1], tpcf[2]), 0) + 1
            terms[(tpcf[2], tpcf[0])] = terms.get((tpcf[2], tpcf[0]), 0) + 1
            terms[(tpcf[0],)] = terms.get((tpcf[0],), 0) - 1
            terms[(tpcf[1],)] = terms.get((tpcf[1],), 0) - 1
            terms[(tpcf[2],)] = terms.get((tpcf[2],), 0) - 1

        edges = sorted(edges)

        # name of the problem
        self.problem = "mono1in3sat"
        # 3SAT formula
        self.cf_ini = cf + 1
        # 1in3SAT formula
        cf = []
        for i, j, k in self.cf_ini.tolist():
            cf.append((i, j, k))
            cf.append((i, -j, -k))
            cf.append((-i, -j, k))
            cf.append((-i, j, -k))
            cf.append((-i, -j, -k))
        self.cf = np.array(cf)
        # edges of the ising graph
        self.edges = np.array(edges)
        # number of variables
        self.numnodes = int(3 * numvar / 2)
        # dictionary of edges of the ising graph
        self.terms = terms

    def cnf_view(self):
        """CNF formula view."""

        numcau = len(self.cf_ini)
        numvar = np.array([abs(np.array(list(i))).max() for i in self.cf_ini]).max()

        adj_mat = np.zeros([numcau, numvar], int)
        for i, r in enumerate(self.cf_ini):
            adj_mat[i][abs(np.array(r)) - 1] = 1

        graph = nx.Graph()

        # add nodes with the bipartite attribute
        graph.add_nodes_from(
            range(numcau),
            bipartite=0,
            color="blue",
            label={i: f"{i+1}" for i in range(numvar)},
        )  # variable nodes
        graph.add_nodes_from(
            range(numcau, numcau + numvar),
            bipartite=1,
            color="red",
            label={i + numvar: f"{i+1}" for i in range(numcau)},
        )  # clause nodes

        # Add edges based on the biadjacency matrix
        for i in range(numcau):
            for j in range(numvar):
                if adj_mat[i][j] == 1:
                    graph.add_edge(j, numvar + i)  # Connect clauses to variables

        pos = nx.spring_layout(graph)  # Generate position for each node
        colors = [
            graph.nodes[n]["color"] for n in graph.nodes
        ]  # Color based on the node attribute

        # Draw nodes and edges
        nx.draw(graph, pos, node_color=colors, with_labels=False, node_size=700)

        # Draw custom labels
        custom_labels = {}
        for node in graph.nodes(data=True):
            custom_labels[node[0]] = (
                node[1]["label"][node[0]] if "label" in node[1] else node[0]
            )

        nx.draw_networkx_labels(graph, pos, labels=custom_labels)

    def ising_view(self):
        """
        Ising formulation graph
        """
        nx.Graph(list(self.edges))


class Mono2satGraph:
    """
    This class instantiates a random bicubic graph representating a monotone 2-SAT problem using qecstruct. It then maps the bicubic graph to an Ising graph using the Ising formulation of the monotone 2-SAT problem.

    Parameters
    ----------
    numvar : int
        Number of variables.

    Attributes
    ----------
    problem : str
        Name of the problem.
    numnodes : int
        Number of variables.
    cf : numpy.ndarray
        2SAT formula.
    edges : numpy.ndarray
        Edges of the Ising graph.
    terms : dict[Tuple[int, int], int]
        Couplings of the Ising graph, where the keys are the edges of the graph and the values are the weights.
    """

    def __init__(self, numvar):
        cg = ig.Graph.Degree_Sequence([3] * numvar, method="vl")
        edgelist = cg.get_edgelist()
        cf = np.array(edgelist) + 1

        terms = {}
        for i, j in edgelist:
            terms[(i,)] = terms.get((i,), 0) + 1
            terms[(j,)] = terms.get((j,), 0) + 1
            terms[(i, j)] = terms.get((i, j), 0) + 1

        # 3SAT formula
        self.cf = cf
        # edges of the ising graph
        self.edges = np.array(edgelist)
        # number of variables
        self.numnodes = numvar
        # dictionary of edges of the ising graph
        self.terms = terms

    def cnf_view(self):
        """CNF formula view."""

        numcau = len(self.cf_ini)
        numvar = np.array([abs(np.array(list(i))).max() for i in self.cf_ini]).max()

        adj_mat = np.zeros([numcau, numvar], int)
        for i, r in enumerate(self.cf_ini):
            adj_mat[i][abs(np.array(r)) - 1] = 1

        graph = nx.Graph()

        # add nodes with the bipartite attribute
        graph.add_nodes_from(
            range(numcau),
            bipartite=0,
            color="blue",
            label={i: f"{i+1}" for i in range(numvar)},
        )  # variable nodes
        graph.add_nodes_from(
            range(numcau, numcau + numvar),
            bipartite=1,
            color="red",
            label={i + numvar: f"{i+1}" for i in range(numcau)},
        )  # clause nodes

        # Add edges based on the biadjacency matrix
        for i in range(numcau):
            for j in range(numvar):
                if adj_mat[i][j] == 1:
                    graph.add_edge(j, numvar + i)  # Connect clauses to variables

        pos = nx.spring_layout(graph)  # Generate position for each node
        colors = [
            graph.nodes[n]["color"] for n in graph.nodes
        ]  # Color based on the node attribute

        # Draw nodes and edges
        nx.draw(graph, pos, node_color=colors, with_labels=False, node_size=700)

        # Draw custom labels
        custom_labels = {}
        for node in graph.nodes(data=True):
            custom_labels[node[0]] = (
                node[1]["label"][node[0]] if "label" in node[1] else node[0]
            )

        nx.draw_networkx_labels(graph, pos, labels=custom_labels)

    def ising_view(self):
        """Ising formulation graph"""

        nx.Graph((list(self.edges)))
