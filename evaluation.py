import numpy as np 
import pandas as pd 
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, matthews_corrcoef
import igraph as ig
import networkx as nx
import graph_tool.all as gt

algorithms = ["bayesian", "spectral", "leiden", "louvain"]
possible_indexes = ["adjusted_mutual_info_score", "adjusted_rand_score", "matthews_corrcoef"]

def compute_indexes(index, algorithm, graphs):
    """Compute a given index for a dictionary of graphs

    Args:
        index (function): the index/score to compute, e.g., adjusted_mutual_info_score
        graphs (dict()): dictionary of the graphs (ig.Graph) we want to study
        algorithm (function): algorithm we want to apply on each graphs

    Returns:
        ind (dict()): return a dictionary of the wanted index for each graph
    """
    assert algorithm in algorithms, "the algorithm must be ones that we implemented"
    assert index in possible_indexes
    
    index_functions = {
        "adjusted_mutual_info_score": adjusted_mutual_info_score,
        "adjusted_rand_score": adjusted_rand_score,
        "matthews_corrcoef": matthews_corrcoef
    }
    compute_index = index_functions[index] 
    
    output = dict()
    
    print(f"Compute the score {index}")
    print(f"Algorithm: {algorithm}")
    
    for key, graph in graphs.items():
        true_clusters = graph.vp["block"].a
        
        #useful for graph conversion
        graph_edges = [(int(e.source()), int(e.target())) for e in graph.edges()]
        
        if algorithm == "bayesian":
            from statistical import bayesianInf
            clusters = bayesianInf(graph)
            predicted_clusters = clusters.a
        
        elif algorithm == "spectral":
            from spectral import spectral
            # Need to convert gt.Graph into nx.Graph
            graph_nx = nx.Graph()
            node_mapping = {v: i for i, v in enumerate(graph.vertices())}
            graph_nx.add_nodes_from(node_mapping.values())
            graph_nx.add_edges_from([(node_mapping[int(e.source())], node_mapping[int(e.target())]) for e in graph.edges()])
            ## Need the graph to be connected...
            if not nx.is_connected(graph_nx):
                print(f"Warning: {key} is not fully connected. Use the largest connected component.")
                lcc = max(nx.connected_components(graph_nx), key=len)
                graph_nx = graph_nx.subgraph(lcc).copy()
                
                node_map = {node: i for i, node in enumerate(graph_nx.nodes())}
                true_clusters = np.array([true_clusters[node] for node in graph_nx.nodes()])
            else:
                node_map = {node: i for i, node in enumerate(graph_nx.nodes())}
                
            predicted_clusters = spectral(graph_nx, K=len(set(true_clusters)))
        
        else:
            # Need to convert gt.Graph into ig.Graph
            graph_ig = ig.Graph()
            graph_ig.add_vertices(n=graph.num_vertices())
            graph_ig.add_edges(graph_edges)
            
            from modularity_based import louvain, leiden
            if algorithm == "louvain":
                clusters = louvain(graph_ig)
            else:
                clusters = leiden(graph_ig)
            predicted_clusters = clusters.membership
        
        if len(true_clusters) != len(predicted_clusters):
            raise ValueError(f"Size mismatch: True labels ({len(true_clusters)}) vs. Predicted labels ({len(predicted_clusters)}) in {key}")
        
        ind = compute_index(true_clusters, predicted_clusters)
        output[key] = ind
        #print(f"{key}: Score = {ind:.5f}")
    avg_ind = np.mean(list(output.values()))
    print(f"\n Average score = {avg_ind}. \n")
    return ind, avg_ind


def validation(K: int=2, nb_probas: int=5):
    from graph_generation import sbm_generation
    sbm_graphs = sbm_generation(K=K, nb_probas=nb_probas)
    results = pd.DataFrame(index=possible_indexes, columns=algorithms)
    for algorithm in algorithms:
        for ind in possible_indexes:
            _, avg_score = compute_indexes(ind, algorithm, sbm_graphs)
            results.loc[ind, algorithm] = avg_score
    return results
