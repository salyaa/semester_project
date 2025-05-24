import numpy as np
import pandas as pd 
import igraph as ig
import networkx as nx
import graph_tool.all as gt
import scipy as sp
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, v_measure_score, homogeneity_score, fowlkes_mallows_score

from algorithms.statistical      import bayesianInf
from algorithms.spectral         import spectral
from algorithms.modularity_based import louvain, leiden
from algorithms.others           import walktrap
from algorithms.spectralClustering import (
    spectralClustering_bm,
    spectralClustering_dcbm,
    spectralClustering_pabm,
    orthogonalSpectralClustering
)
from algorithms.selfrepresentation import *

from constants import *

_metric_fns = {
    "adjusted_mutual_info_score": adjusted_mutual_info_score,
    "adjusted_rand_score":        adjusted_rand_score,
    "v_measure_score":            v_measure_score,
    "homogeneity_score":          homogeneity_score,
    "fowlkes_mallows_score":      fowlkes_mallows_score,
}

def adjacency_matrix_of(G):
    """Return an (n×n) SciPy sparse matrix for igraph, networkx or graph-tool."""
    if isinstance(G, ig.Graph):
        rows, cols = zip(*G.get_edgelist())
        data = np.ones(len(rows))
        return sp.sparse.csr_matrix((data, (rows, cols)), shape=(G.vcount(), G.vcount()))
    elif isinstance(G, nx.Graph):
        return nx.to_scipy_sparse_matrix(G, format="csr")
    else:  # graph-tool
        n = G.num_vertices()
        rows, cols = [], []
        for e in G.edges():
            rows.append(int(e.source())); cols.append(int(e.target()))
        data = np.ones(len(rows))
        return sp.sparse.csr_matrix((data, (rows, cols)), shape=(n, n))


def get_predicted_memberships(algorithm, graphs, block_membership, possible_algos=algorithms, graph_type="sbm"):
    """
    Run each `graphs[name]` through your chosen algorithm,
    return { name: np.array(membership) }.
    """
    if possible_algos is None:
        possible_algos = algorithms
    assert algorithm in possible_algos, "The algorithm must be one that we implemented."
    assert graph_type in graph_types, "The type of graph must be one that we implemented."
    
    predictions = {}
    trues = {}
    
    for name, G in graphs.items():
        true_cluster = block_membership[name]
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
                gtG = G  # already a graph-tool graph
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
                true_cluster = np.array([true_cluster[node] for node in nxg.nodes()])
            pred = spectral(nxg, K=len(set(true_cluster))) 

        elif algorithm in ("louvain","leiden"):
            if not isinstance(G, ig.Graph):
                igG = ig.Graph()
                igG.add_vertices(n=G.num_vertices())
                igG.add_edges(graph_edges)
                G = igG
            method = leiden if algorithm=="leiden" else louvain
            clusters = method(G)
            pred = np.array(clusters.membership)

        elif algorithm == "walktrap":  # walktrap
            clusters = walktrap(G, K=len(set(true_cluster)))
            pred = np.array(clusters.membership)
        
        else: # the 4 new spectral algorithms
            A = adjacency_matrix_of(G)
            spectral_map = {
                "sc_bm":    spectralClustering_bm,
                "sc_dcbm":  spectralClustering_dcbm,
                "sc_pabm":  spectralClustering_pabm,
                "orth_sc":  orthogonalSpectralClustering
            }
            fn = spectral_map.get(algorithm)
            if fn is None:
                raise ValueError(f"Unknown spectral variant: {algorithm}")
            try:
                raw = fn(A, n_clusters=len(set(true_cluster)))
            except TypeError as e:
                msg = str(e)
                if "k >= N" in msg or ">= N" in msg:
                    # convert to dense and retry
                    A2 = A.toarray() if hasattr(A, "toarray") else np.array(A)
                    raw = fn(A2, n_clusters=len(set(true_cluster)))
                else:
                    raise
            pred = np.array(raw, dtype=int) - 1
        
        if len(true_cluster) != len(pred):
            raise ValueError(f"Size mismatch: True labels ({len(true_cluster)}) vs. Predicted labels ({len(pred)}) in {name}")
        
        predictions[name] = pred
        trues[name] = true_cluster
    
    return predictions, trues

def evaluate_all(graphs, block_membership, possible_algos=None, graph_type: str="sbm"):
    rows = []
    for algo in possible_algos:
        preds, trues = get_predicted_memberships(algo, graphs, block_membership, possible_algos=possible_algos, graph_type=graph_type)
        for metric in possible_metrics:
            fn = _metric_fns[metric]
            scores = []
            nfound = []
            for name in graphs:
                t = trues[name]
                p = preds[name]
                scores.append(fn(t, p))
                nfound.append(len(np.unique(p)))
            rows.append({
                "algorithm": algo,
                "metric": metric,
                "scores": scores,
                "mean": np.mean(scores),
                "std": np.std(scores),
                "nfound": np.mean(nfound),
                "mean_n": np.mean(nfound),
            })
    df = pd.DataFrame(rows)
    df.set_index(["algorithm", "metric"], inplace=True)
    return df