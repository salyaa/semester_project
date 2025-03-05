import numpy as np 
import pandas as pd 
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, v_measure_score
import igraph as ig
import networkx as nx
import graph_tool.all as gt
import matplotlib.pyplot as plt

from graph_generation import sbm_generation

algorithms = ["bayesian", "spectral", "leiden", "louvain"]
possible_indexes = ["adjusted_mutual_info_score", "adjusted_rand_score", "v_measure_score"]
# Other possible indices:
#   - homogeneity_score
#   - fowlkes_mallow_score


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
        "v_measure_score": v_measure_score #identical to normalized_mutual_info_score with the 'arithmetic' option for averaging.
    }
    compute_index = index_functions[index] 
    
    output = dict()
    #print(f"Compute the score {index} for {algorithm}")
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
            ## Need to convert gt.Graph into nx.Graph
            graph_nx = nx.Graph()
            node_mapping = {v: i for i, v in enumerate(graph.vertices())}
            graph_nx.add_nodes_from(node_mapping.values())
            graph_nx.add_edges_from([(node_mapping[int(e.source())], node_mapping[int(e.target())]) for e in graph.edges()])
            ## Need the graph to be connected... in case it is not do this:
            if not nx.is_connected(graph_nx):
                #print(f"Warning: {key} is not fully connected. Use the largest connected component.")
                lcc = max(nx.connected_components(graph_nx), key=len)
                graph_nx = graph_nx.subgraph(lcc).copy()
                
                node_map = {node: i for i, node in enumerate(graph_nx.nodes())}
                true_clusters = np.array([true_clusters[node] for node in graph_nx.nodes()])
            else:
                node_map = {node: i for i, node in enumerate(graph_nx.nodes())}
                
            predicted_clusters = spectral(graph_nx, K=len(set(true_clusters)))
        
        else:
            ## Need to convert gt.Graph into ig.Graph
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
    #print(f"\n Average score = {avg_ind}. \n")
    
    return ind, avg_ind


def validation(K: int=2, nb_probas: int=5, modify : str="out"):
    """Complete the average score for all the possible pairs (index, algorithm)

    Args:
        K (int, optional): Number of clusters in our generated graph. Defaults to 2.
        nb_probas (int, optional): Number of graphs. Defaults to 5.
        modify (str, optional): Decide if we modify p_in or p_out. Defaults to "out".

    Returns:
        results (pd.DataFrame): Dataframe representing the average scores for each pairs.
    """
    sbm_graphs = sbm_generation(K=K, nb_probas=nb_probas, modify=modify)
    results = pd.DataFrame(index=possible_indexes, columns=algorithms)
    for algorithm in algorithms:
        for ind in possible_indexes:
            _, avg_score = compute_indexes(ind, algorithm, sbm_graphs)
            results.loc[ind, algorithm] = avg_score
    print(f"Average scores computed for K = {K} and {nb_probas} graphs!")
    
    ## Save the results in a csv file
    name_table = f"evaluations/scores_K{K}_n{nb_probas}.csv"
    results.to_csv(name_table, index=True)
    print("\n Results successfully saved!")
    
    return results


def validation_range_K(range_K, nb_probas: int=5, modify: str="out", plot: bool=False):
    """Complete the average score for all possible pairs (index, algorithm) for various number of true clusters.

    Args:
        range_K (np.array): The different values of clusters we consider.
        nb_probas (int, optional): Number of graphs. Defaults to 5.
        modify (str, optional): Decide if we modify p_in or p_out. Defaults to "out".
        plot (boolean, optional): if True, plot the results over range_K for each index. Defaults to False.

    Returns:
        results (pd.DataFrame): Dataframe with multiple indexes (K, index) representing the average score for each pair.
    """
    multi_ind = pd.MultiIndex.from_product(
        [range_K, possible_indexes], names=["K", "Metric"]
    )
    results = pd.DataFrame(index=multi_ind, columns=algorithms)
    
    for K in range_K:
        sbm_graphs = sbm_generation(K=K, nb_probas=nb_probas, modify=modify)
        for algorithm in algorithms:
            for index in possible_indexes:
                _, avg_score = compute_indexes(index, algorithm, sbm_graphs)
                results.loc[(K, index), algorithm] = avg_score
        print(f"Completed K = {K}.")
    
    ## Save the results in a csv file
    name_table = f"evaluations/scores_K{range_K[0]}to{range_K[-1]}_n{nb_probas}_p{modify}.csv"
    results.to_csv(name_table)
    print(f"Results saved at {name_table}!")
    
    if plot:
        title = f"Community detection performance across different scores, using {nb_probas} SBM graphs modifying p_{modify}"
        plot_results_range_K(results, title)
    
    return results


def plot_results_range_K(results, title: str=None):
    algos = results.columns 
    indexes = results.index.get_level_values("Metric").unique()
    range_K = results.index.get_level_values("K").unique()  

    f, axs = plt.subplots(1, len(indexes), figsize=(5 * len(indexes), 5), sharex=True, sharey=True)

    if len(indexes) == 1: # if there is only one index
        axs = [axs]

    for i, ind in enumerate(indexes):
        avg_scores = results.xs(ind, level="Metric")

        for algo in algos:
            axs[i].plot(range_K, avg_scores[algo], marker='x', label=algo)

        # Plot settings
        axs[i].set_title(ind.replace("_", " ").title())
        axs[i].set_xlabel("Number of Clusters (K)")
        axs[i].set_ylabel("Score")
        axs[i].legend()
        axs[i].grid(True)
        
    f.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
