import numpy as np 
import pandas as pd 
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, v_measure_score, homogeneity_score, fowlkes_mallows_score
import igraph as ig
import networkx as nx
import graph_tool.all as gt
import matplotlib.pyplot as plt

from graph_generation import sbm_generation


algorithms = ["bayesian", "bayesian_fixed_K", "spectral", "leiden", "louvain"]
possible_metrics = [
    "adjusted_mutual_info_score", 
    "adjusted_rand_score", 
    "v_measure_score",
    "homogeneity_score",
    "fowlkes_mallows_score"
]


def compute_score(metric, algorithm, graphs):
    """Compute a given index for a dictionary of graphs

    Args:
        index (function): the index/score to compute, e.g., adjusted_mutual_info_score
        graphs (dict()): dictionary of the graphs (ig.Graph) we want to study
        algorithm (function): algorithm we want to apply on each graphs

    Returns:
        ind (dict()): return a dictionary of the wanted index for each graph
    """
    assert algorithm in algorithms, "The algorithm must be ones that we implemented."
    assert metric in possible_metrics
    
    metric_functions = {
        "adjusted_mutual_info_score": adjusted_mutual_info_score,
        "adjusted_rand_score": adjusted_rand_score,
        "v_measure_score": v_measure_score, #identical to normalized_mutual_info_score with the 'arithmetic' option for averaging.
        "homogeneity_score": homogeneity_score,
        "fowlkes_mallows_score": fowlkes_mallows_score
    }
    compute_metric = metric_functions[metric] 
    
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
        elif algorithm == "bayesian_fixed_K":
            from statistical import bayesianInfFixedK
            clusters = bayesianInfFixedK(graph, len(set(true_clusters)))
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
                
                true_clusters = np.array([true_clusters[node] for node in graph_nx.nodes()])
            
            predicted_clusters = spectral(graph_nx, K=len(set(true_clusters)))
        
        else:
            ## Need to convert gt.Graph into ig.Graph
            graph_ig = ig.Graph()
            graph_ig.add_vertices(n=graph.num_vertices())
            graph_ig.add_edges(graph_edges)
            
            from modularity_based import louvain, leiden
            if algorithm == "louvain":
                clusters = louvain(graph_ig)
            else: # algorithm == "leiden"
                clusters = leiden(graph_ig)
            predicted_clusters = clusters.membership
        
        # Had issue with graph conversion, so test this:
        if len(true_clusters) != len(predicted_clusters):
            raise ValueError(f"Size mismatch: True labels ({len(true_clusters)}) vs. Predicted labels ({len(predicted_clusters)}) in {key}")
        
        score = compute_metric(true_clusters, predicted_clusters)
        output[key] = score
        #print(f"{key}: Score = {ind:.5f}")
    avg_score = np.mean(list(output.values()))
    #print(f"\n Average score = {avg_ind}. \n")
    std_score = np.std(list(output.values()))
    
    return score, avg_score, std_score


def validation(K: int=2, nb_probas: int=5, modify : str="out"):
    """Complete the average score for all the possible pairs (index, algorithm)

    Args:
        K (int, optional): Number of clusters in our generated graph. Defaults to 2.
        nb_probas (int, optional): Number of graphs. Defaults to 5.
        modify (str, optional): Decide if we modify p_in or p_out. Defaults to "out".

    Returns:
        results (pd.DataFrame): Dataframe representing the average scores for each pairs.
    """
    sbm_graphs, _ = sbm_generation(K=K, nb_probas=nb_probas, modify=modify)
    results = pd.DataFrame(index=possible_metrics, columns=algorithms)
    for algorithm in algorithms:
        for metric in possible_metrics:
            _, avg_score, _ = compute_score(metric, algorithm, sbm_graphs)
            results.loc[metric, algorithm] = avg_score
    print(f"Average scores computed for K = {K} and {nb_probas} graphs!")
    
    ## Save the results in a csv file
    name_table = f"evaluations/scores_K{K}_n{nb_probas}.csv"
    results.to_csv(name_table, index=True)
    print("\n Results successfully saved!")
    
    return results


def validation_range_K(range_K, modify: str="out", plot: bool=True):
    """Compute the score for all possible pairs (index, algorithm) for various number of true communities, tested on a single graph.

    Args:
        range_K (np.array): The different values of clusters we consider.
        nb_probas (int, optional): Number of graphs. Defaults to 5.
        modify (str, optional): Decide if we modify p_in or p_out. Defaults to "out".
        plot (boolean, optional): if True, plot the results over range_K for each score. Defaults to True.

    Returns:
        results (pd.DataFrame): Dataframe with multiple indexes (K, index) representing the average score for each pair.
    """
    multi_ind = pd.MultiIndex.from_product(
        [range_K, possible_metrics], names=["K", "Metric"]
    )
    results_mean = pd.DataFrame(index=multi_ind, columns=algorithms)
    results_std = pd.DataFrame(index=multi_ind, columns=algorithms)
    
    for K in range_K:
        sbm_graphs, _ = sbm_generation(K=K, nb_probas=1, modify=modify)
        for algorithm in algorithms:
            for metric in possible_metrics:
                _, avg_score, std_score = compute_score(metric, algorithm, sbm_graphs)
                results_mean.loc[(K, metric), algorithm] = avg_score
                results_std.loc[(K, metric), algorithm] = std_score
        print(f"Completed K = {K}.")
    
    ## Save the results in a csv file
    name_table = f"evaluations/scores_K{range_K[0]}to{range_K[-1]}_p{modify}.csv"
    results_mean.to_csv(name_table)
    print(f"Results saved at {name_table}!")
    
    if plot:
        title = f"Community detection performance across different scores, using a SBM graph for various number of cluster K"
        plot_results_range_K(results_mean, results_std, title)
    
    return results_mean


def validation_range_p(range_p: np.array=None, K: int=3, p: float=0.5, modify: str="out", plot: bool=True):
    """Compute the average score for all possible pairs (index, algorithm) for a fixed number of true communities K, tested for various SBM graphs modifying p_in or p_out depending on modify.

    Args:
        range_p (np.array, optional): The different probability values we want to test out. Defaults to None.
        K (int, optional): Number of true communities to consider. Defaults to 3.
        p (float, optional): Fixed probabilities of the SBM graphs. Defaults to 0.5.
        modify (str, optional): Define if we modify p_in or p_out. Defaults to "out".
        plot (bool, optional): If True, plot the results over range_p for each score. Defaults to True.

    Returns:
        results (pd.DataFrame): Dataframe with multiple indexes (p, index) representing the average score for each pair.
    """
    sbm_graphs, proba_range = sbm_generation(K=K, p=p, range_p=range_p, modify=modify)
    
    multi_ind = pd.MultiIndex.from_product(
        [proba_range, possible_metrics], names=[f"p_{modify}", "Metric"]
    )
    results = pd.DataFrame(index=multi_ind, columns=algorithms)
    results_std = pd.DataFrame(index=multi_ind, columns=algorithms)
    
    for p in proba_range:
        for algorithm in algorithms:
            for metric in possible_metrics:
                _, avg_score, std_score = compute_score(metric, algorithm, sbm_graphs)
                results.loc[(p, metric), algorithm] = avg_score
                results_std.loc[(p, metric), algorithm] = std_score
        print(f"Completed p_{modify} = {p}.")
    
    ## Save the results in a csv file
    name_table = f"evaluations/scores_K{K}_n{len(proba_range)}_p{modify}.csv"
    results.to_csv(name_table)
    print(f"Results saved at {name_table}!")
    
    if plot:
        title = f"Community detection average performance for {K} communities across different scores, using {len(proba_range)} SBM graphs"
        plot_results_range_p(results, results_std, modify, title)
    
    return results


def plot_results_range_K(results_mean, results_std, title: str=None):
    algos = results_mean.columns 
    metrics = results_mean.index.get_level_values("Metric").unique()
    range_K = results_mean.index.get_level_values("K").unique()  

    f, axs = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5), sharex=True, sharey=True)

    if len(metrics) == 1: 
        axs = [axs]

    for i, ind in enumerate(metrics):
        avg_scores = results_mean.xs(ind, level="Metric")ß
        std_scores = results_std.xs(ind, level="Metric")

        for algo in algos:
            mean_values = avg_scores[algo].astype(float)
            std_values = std_scores[algo].astype(float)
            
            std_values = np.nan_to_num(std_values, nan=0.0)

            axs[i].plot(range_K, mean_values, marker='o', label=algo)
            axs[i].fill_between(range_K, mean_values - std_values, mean_values + std_values, alpha=0.2)

        axs[i].set_title(ind.replace("_", " ").title())
        axs[i].set_xlabel("Number of Clusters (K)")
        axs[i].set_ylabel("Score")
        axs[i].legend()
        axs[i].grid(True)
        
    f.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()



def plot_results_range_p(results_mean, results_std, modify: str="out", title: str=None):
    algos = results_mean.columns 
    metrics = results_mean.index.get_level_values("Metric").unique()
    range_p = results_mean.index.get_level_values(f"p_{modify}").unique()  

    f, axs = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5), sharex=True, sharey=True)

    if len(metrics) == 1: 
        axs = [axs]

    for i, metric in enumerate(metrics):
        avg_scores = results_mean.xs(metric, level="Metric")
        std_scores = results_std.xs(metric, level="Metric")

        for algo in algos:
            mean_values = avg_scores[algo].astype(float) 
            std_values = std_scores[algo].astype(float)  

            std_values = np.nan_to_num(std_values, nan=0.0) 

            axs[i].plot(range_p, mean_values, marker='o', label=algo)
            axs[i].fill_between(range_p, mean_values - std_values, mean_values + std_values, alpha=0.2)

        axs[i].set_title(metric.replace("_", " ").title())
        axs[i].set_xlabel("Probability p")
        axs[i].set_ylabel("Score")
        axs[i].legend()
        axs[i].grid(True)
        
    f.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
    