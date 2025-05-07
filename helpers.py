import numpy as np
import igraph as ig
import networkx as nx
import graph_tool.all as gt

from algorithms.statistical      import bayesianInf
from algorithms.spectral         import spectral
from algorithms.modularity_based import louvain, leiden
from algorithms.others           import walktrap


def get_predicted_memberships(algorithm, graphs, block_membership, graph_type="sbm"):
    """
    Run each `graphs[name]` through your chosen algorithm,
    return { name: np.array(membership) }.
    """
    memberships = {}
    for name, G in graphs.items():
        true_clusters = block_membership[name]
        if isinstance(G, gt.Graph):
            graph_edges = [(int(e.source()), int(e.target())) for e in G.edges()]
        else: 
            graph_edges = G.get_edgelist()
        # convert to graph-tool if needed for bayesian:
        if algorithm == "bayesian":
            gtG = gt.Graph(directed=False)
            if isinstance(G, ig.Graph):
                gtG.add_vertex(n=G.vcount())
                gtG.add_edge_list(graph_edges)
            else:
                gtG = G  # assume it's already graph‐tool
            clusters = bayesianInf(gtG, deg_corr=(graph_type=="abcd"))
            pred = clusters.a

        elif algorithm == "spectral":
            # to networkx
            nxg = nx.Graph()
            if isinstance(G, ig.Graph):
                node_mapping = {v.index: i for i, v in enumerate(G.vs)}
                nxg.add_nodes_from(node_mapping.values())
                nxg.add_edges_from([(node_mapping[u], node_mapping[v]) for u, v in G.get_edgelist()])
            else:
                node_mapping = {v: i for i, v in enumerate(G.vertices())}
                nxg.add_nodes_from(node_mapping.values())
                nxg.add_edges_from([(node_mapping[int(e.source())], node_mapping[int(e.target())]) for e in G.edges()])
            if not nx.is_connected(nxg):
                lcc = max(nx.connected_components(nxg), key=len)
                nxg = nxg.subgraph(lcc).copy()
                true_clusters = np.array([true_clusters[node] for node in nxg.nodes()])
            pred = spectral(nxg, K=len(set(true_clusters)))  # your spectral infers K or use true K

        elif algorithm in ("louvain","leiden"):
            if not isinstance(G, ig.Graph):
                igG = ig.Graph()
                igG.add_vertices(G.num_vertices())
                igG.add_edges(graph_edges)
                G = igG
            method = leiden if algorithm=="leiden" else louvain
            clusters = method(G)
            pred = np.array(clusters.membership)

        else:  # walktrap
            clusters = walktrap(G, K=len(set(true_clusters)))
            pred = np.array(clusters.membership)

        memberships[name] = np.array(pred, dtype=int)

    return memberships
