import numpy as np
import pandas as pd 
import igraph as ig
import networkx as nx
import graph_tool.all as gt

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

from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, v_measure_score, homogeneity_score, fowlkes_mallows_score

from constants import *

metric_functions = {
        "adjusted_mutual_info_score": adjusted_mutual_info_score,
        "adjusted_rand_score": adjusted_rand_score,
        "v_measure_score": v_measure_score,
        "homogeneity_score": homogeneity_score,
        "fowlkes_mallows_score": fowlkes_mallows_score
    }


def get_predicted_memberships(algorithm, graphs, block_membership, possible_algos=None, graph_type="sbm"):
    """
    Run each `graphs[name]` through your chosen algorithm,
    return { name: np.array(membership) } and true memberships.
    """
    if possible_algos is None:
        possible_algos = algorithms
    assert algorithm in possible_algos, "The algorithm must be one that we implemented."
    assert graph_type in ["sbm", "abcd"], "The type of graph must be 'sbm' or 'abcd'."
    
    predictions = {}
    trues = {}
    for name, G in graphs.items():
        true_cluster = block_membership[name]
        # convert to edge list
        if isinstance(G, gt.Graph):
            edge_list = [(int(e.source()), int(e.target())) for e in G.edges()]
        else:
            edge_list = G.get_edgelist()

        # choose algorithm
        if algorithm == "bayesian":
            # convert to graph-tool
            if isinstance(G, ig.Graph):
                gtG = gt.Graph(directed=False)
                gtG.add_vertex(n=G.vcount())
                gtG.add_edge_list(edge_list)
                deg_correction = graph_type == "abcd"
            clusters = bayesianInf(gtG, deg_corr=deg_correction)
            pred = clusters.a

        elif algorithm == "spectral":
            # build networkx
            nxg = nx.Graph()
            if isinstance(G, ig.Graph):
                node_mapping = {v.index: i for i, v in enumerate(G.vs)}
                nxg.add_nodes_from(node_mapping.values())
                nxg.add_edges_from([(node_mapping[u], node_mapping[v]) for u, v in edge_list])
            else:
                node_mapping = {v: i for i, v in enumerate(G.vertices())}
                nxg.add_nodes_from(node_mapping.values())
                nxg.add_edges_from([(node_mapping[int(e.source())], node_mapping[int(e.target())]) for e in G.edges()])
            if not nx.is_connected(nxg):
                lcc = max(nx.connected_components(nxg), key=len)
                nxg = nxg.subgraph(lcc).copy()
                true_cluster = np.array([true_cluster[node] for node in nxg.nodes()])
            pred = spectral(nxg, K=len(set(true_cluster)))

        elif algorithm in ("louvain", "leiden"):
            # convert to igraph
            if not isinstance(G, ig.Graph):
                igG = ig.Graph()
                igG.add_vertices(n=G.num_vertices())
                igG.add_edges(edge_list)
                G = igG
            method = leiden if algorithm == "leiden" else louvain
            clusters = method(G)
            pred = clusters.membership

        elif algorithm == "walktrap":
            clusters = walktrap(G, K=len(set(true_cluster)))
            pred = clusters.membership

        else: # custom spectral variants
            #A = adjacency_matrix_of(G)
            if not isinstance(G, ig.Graph):
                igG = ig.Graph()
                igG.add_vertices(n=G.num_vertices())
                igG.add_edges(edge_list)
                G = igG
            A = G.get_adjacency_sparse()
            spec_map = {
                "sc_bm":   spectralClustering_bm,
                "sc_dcbm": spectralClustering_dcbm,
                "sc_pabm": spectralClustering_pabm,
                "orth_sc": orthogonalSpectralClustering
            }
            fn = spec_map.get(algorithm)
            if fn is None:
                raise ValueError(f"Unknown spectral variant: {algorithm}")
            # delegate entirely to the spectralClustering function
            pred = fn(A, n_clusters=len(set(true_cluster)))

        # sanity check
        if len(true_cluster) != len(pred):
            raise ValueError(
                f"Size mismatch: true={len(true_cluster)} vs pred={len(pred)} in {name}"
            )
        predictions[name] = np.array(pred, dtype=int)
        trues[name] = np.array(true_cluster, dtype=int)

    return predictions, trues


def evaluate_all(graphs, block_membership, possible_algos=None, graph_type: str="sbm"):
    rows = []
    for algo in possible_algos:
        preds, trues = get_predicted_memberships(algo, graphs, block_membership, possible_algos=possible_algos, graph_type=graph_type)
        for metric in possible_metrics:
            fn = metric_functions[metric]
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
                "mean": np.mean(list(scores)),
                "std": np.std(list(scores)),
                "nfound": np.mean(nfound),
                "mean_n": np.mean(nfound),
            })
    df = pd.DataFrame(rows)
    df.set_index(["algorithm", "metric"], inplace=True)
    return df