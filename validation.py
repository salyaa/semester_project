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
        score (dict()): return a dictionary of the wanted score for each graph
        avg_score (float): return the average score
        std_score (float): return the standard deviation
        number_clusters_found (int): number of clusters inferred by the algorithm, useful to compare it with the real number of clusters
    """
    assert algorithm in algorithms, "The algorithm must be ones that we implemented."
    assert metric in possible_metrics, "The metric must be one that we selected."
    
    metric_functions = {
        "adjusted_mutual_info_score": adjusted_mutual_info_score,
        "adjusted_rand_score": adjusted_rand_score,
        "v_measure_score": v_measure_score, #identical to normalized_mutual_info_score with the 'arithmetic' option for averaging.
        "homogeneity_score": homogeneity_score,
        "fowlkes_mallows_score": fowlkes_mallows_score
    }
    compute_metric = metric_functions[metric] 
    
    output = dict()
    number_clusters_found = 0
    
    for key, graph in graphs.items():
        true_clusters = graph.vp["block"].a
        #useful for graph conversion
        graph_edges = [(int(e.source()), int(e.target())) for e in graph.edges()]
        
        if algorithm == "bayesian":
            from statistical import bayesianInf
            clusters = bayesianInf(graph)
            predicted_clusters = clusters.a
            number_clusters_found = len(np.unique(predicted_clusters))
        elif algorithm == "bayesian_fixed_K":
            from statistical import bayesianInfFixedK
            clusters = bayesianInfFixedK(graph, len(set(true_clusters)))
            predicted_clusters = clusters.a
            number_clusters_found = len(np.unique(predicted_clusters))
        
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
            number_clusters_found = len(np.unique(predicted_clusters))
        
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
            number_clusters_found = len(np.unique(predicted_clusters))
        
        # Had issue with graph conversion, so test this:
        if len(true_clusters) != len(predicted_clusters):
            raise ValueError(f"Size mismatch: True labels ({len(true_clusters)}) vs. Predicted labels ({len(predicted_clusters)}) in {key}")
        
        score = compute_metric(true_clusters, predicted_clusters)
        output[key] = score
        #print(f"{key}: Score = {ind:.5f}")
    avg_score = np.mean(list(output.values()))
    #print(f"\n Average score = {avg_ind}. \n")
    std_score = np.std(list(output.values()))
    
    return score, avg_score, std_score, number_clusters_found


def validation_range_K(range_K, modify: str="out", plot: bool=True):
    """Compute the score for all possible pairs (index, algorithm) for various number of true communities, tested on a single graph.

    Args:
        range_K (np.array): The different values of clusters we consider.
        nb_probas (int, optional): Number of graphs. Defaults to 5.
        modify (str, optional): Decide if we modify p_in or p_out. Defaults to "out".
        plot (boolean, optional): if True, plot the results over range_K for each score. Defaults to True.

    Returns:
        results (pd.DataFrame): Dataframe with multiple indexes (K, index) representing the average score for each pair.
        number_clusters (pd.DataFrame): Dataframe with the number of clusters found for each algorithm for a given true K.
    """
    multi_ind = pd.MultiIndex.from_product(
        [range_K, possible_metrics], names=["K", "Metric"]
    )
    results_mean = pd.DataFrame(index=multi_ind, columns=algorithms)
    results_std = pd.DataFrame(index=multi_ind, columns=algorithms)
    
    number_clusters = pd.DataFrame(index=range_K, columns=algorithms)
    
    for K in range_K:
        sbm_graphs, _ = sbm_generation(K=K, nb_probas=1, modify=modify)
        for algorithm in algorithms:
            k_algo_avg = []
            for metric in possible_metrics:
                _, avg_score, std_score, k = compute_score(metric, algorithm, sbm_graphs)
                results_mean.loc[(K, metric), algorithm] = avg_score
                results_std.loc[(K, metric), algorithm] = std_score
                k_algo_avg.append(k)
            number_clusters.loc[K, algorithm] = np.mean(k_algo_avg)
        print(f"Completed K = {K}.")
    
    ## Save the results in a csv file
    name_table = f"evaluations/scores_K{range_K[0]}to{range_K[-1]}_p{modify}.csv"
    results_mean.to_csv(name_table)
    print(f"Results saved at {name_table}!")
    
    if plot:
        title = f"Community detection performance across different scores, using a SBM graph for various number of cluster K"
        plot_results_range_K(results_mean, results_std, title, number_clusters)
        plot_inferred_K(number_clusters, range_K)
    
    return results_mean, number_clusters


def validation_range_p(range_p: np.array=None, K: int=3, modify: str="out", plot: bool=True):
    """Compute the average score for all possible pairs (index, algorithm) for a fixed number of true communities K, tested for various SBM graphs modifying p_in or p_out depending on modify.

    Args:
        range_p (np.array, optional): The different probability values we want to test out. Defaults to None.
        K (int, optional): Number of true communities to consider. Defaults to 3.
        modify (str, optional): Define if we modify p_in or p_out. Defaults to "out".
        plot (bool, optional): If True, plot the results over range_p for each score. Defaults to True.

    Returns:
        results (pd.DataFrame): Dataframe with multiple indexes (p, index) representing the average score for each pair.
        number_clusters (pd.DataFrame): Dataframe with the number of clusters found for each algorithm for a given true K, for each probability value we considered.
    """
    sbm_graphs, proba_range = sbm_generation(K=K, range_p=range_p, modify=modify)
    
    multi_ind = pd.MultiIndex.from_product(
        [proba_range, possible_metrics], names=[f"p_{modify}", "Metric"]
    )
    results = pd.DataFrame(index=multi_ind, columns=algorithms)
    results_std = pd.DataFrame(index=multi_ind, columns=algorithms)
    
    number_clusters = pd.DataFrame(index=range_p, columns=algorithms)
    
    for p in proba_range:
        for algorithm in algorithms:
            k_algo_avg = []
            for metric in possible_metrics:
                _, avg_score, std_score, k = compute_score(metric, algorithm, sbm_graphs)
                results.loc[(p, metric), algorithm] = avg_score
                results_std.loc[(p, metric), algorithm] = std_score
                k_algo_avg.append(k)
            number_clusters.loc[p, algorithm] = np.mean(k_algo_avg)
        print(f"Completed p_{modify} = {p}.")
    
    ## Save the results in a csv file
    name_table = f"evaluations/scores_K{K}_n{len(proba_range)}_p{modify}.csv"
    results.to_csv(name_table)
    print(f"Results saved at {name_table}!")
    
    if plot:
        title = f"Community detection average performance for {K} communities across different scores, using {len(proba_range)} SBM graphs"
        plot_results_range_p(results, results_std, modify, title)
        plot_inferred_K_fixed_param(number_clusters, proba_range, f"p_{modify}", K, f"Inferred Clusters vs p_{modify} (K={K})")
    
    return results, number_clusters


def validation_range_n(range_n: np.array=None, K: int=3, modify: str="out", plot: bool=True):
    """Compute the average score for all possible pairs (index, algorithm) for a fixed number of true communities K, tested for various SBM graphs modifying the number of nodes n.

    Args:
        range_n (np.array, optional): The different number of nodes we want to test out. Defaults to None.
        K (int, optional): Number of true communities to consider. Defaults to 3.
        modify (str, optional): Define if we modify p_in or p_out. Defaults to "out".
        plot (bool, optional): If True, plot the results over range_p for each score. Defaults to True.

    Returns:
        results (pd.DataFrame): Dataframe with multiple indexes (p, index) representing the average score for each pair.
        number_clusters (pd.DataFrame): Dataframe with the number of clusters found for each algorithm for a given true K, and for each size of the network we considered.
    """
    if range_n is None:
        range_n = np.linspace(100*K, 1000*K, 10, dtype=np.int32)
    print(range_n)
    
    multi_ind = pd.MultiIndex.from_product(
        [range_n, possible_metrics], names=["n", "Metric"]
    )
    results = pd.DataFrame(index=multi_ind, columns=algorithms)
    results_std = pd.DataFrame(index=multi_ind, columns=algorithms)
    
    number_clusters = pd.DataFrame(index=range_n, columns=algorithms)
    
    for n in range_n:
        sbm_graphs, _ = sbm_generation(n=n, K=K, nb_probas=1, modify=modify)
        for algorithm in algorithms:
            k_algo_avg = []
            for metric in possible_metrics:
                _, avg_score, std_score, k = compute_score(metric, algorithm, sbm_graphs)
                results.loc[(n, metric), algorithm] = avg_score
                results_std.loc[(n, metric), algorithm] = std_score
                k_algo_avg.append(k)
            number_clusters.loc[n, algorithm] = np.mean(k_algo_avg)
        print(f"Completed n = {n}.")
        
    ## Save the results in a csv file
    name_table = f"evaluations/scores_K{K}_n_from{range_n[0]}to{range_n[-1]}_p{modify}.csv"
    results.to_csv(name_table)
    print(f"Results saved at {name_table}!")
    
    if plot:
        title = f"Community detection average performance for {K} communities across different scores, using SBM graphs varying the number of nodes n"
        plot_results_range_n(results, results_std, modify, title)
        plot_inferred_K_fixed_param(
            number_clusters, 
            param_range=range_n, 
            param_name="n", 
            true_K=K,
            title=f"Inferred Clusters vs Network Size n (K={K})"
        )
    
    return results, number_clusters


def plot_results_range_K(results_mean, results_std, title: str=None, number_clusters=None):
    from IPython.display import clear_output
    clear_output(wait=True)
    
    algos = results_mean.columns 
    metrics = results_mean.index.get_level_values("Metric").unique()
    range_K = results_mean.index.get_level_values("K").unique()  

    f, axs = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5), sharex=True, sharey=False)

    if len(metrics) == 1: 
        axs = [axs]

    for i, metric in enumerate(metrics):
        avg_scores = results_mean.xs(metric, level="Metric")
        std_scores = results_std.xs(metric, level="Metric")

        ax = axs[i]
        for algo in algos:
            mean_values = avg_scores[algo].astype(float)
            std_values = std_scores[algo].astype(float)
            std_values = np.nan_to_num(std_values, nan=0.0)

            ax.plot(range_K, mean_values, marker='x', label=algo)
            ax.fill_between(range_K, mean_values - std_values, mean_values + std_values, alpha=0.2)

        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel("Number of Clusters (K)")
        ax.set_ylabel("Score")
        ax.grid(True)

        # # ➕ Add secondary y-axis for number of clusters found
        # if number_clusters is not None:
        #     ax2 = ax.twinx()
        #     for algo in algos:
        #         k_found = number_clusters[algo].astype(float)
        #         ax2.plot(range_K, k_found, linestyle="--", marker='o', label=f"{algo} (K)", alpha=0.6)
        #     ax2.set_ylabel("Inferred Number of Clusters")

    axs[0].legend(loc="upper left")
    f.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

def plot_inferred_K(number_clusters, true_K_range, title="Inferred Number of Clusters"):
    plt.figure(figsize=(10, 6))
    for algo in number_clusters.columns:
        plt.plot(true_K_range, number_clusters[algo].astype(float), marker="o", label=algo)
    plt.plot(true_K_range, true_K_range, "--", label="True K", color="black")
    plt.xlabel("True Number of Clusters (K)")
    plt.ylabel("Inferred Number of Clusters")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_inferred_K_fixed_param(number_clusters, param_range, param_name="p_out", true_K=None, title="Inferred Number of Clusters"):
    plt.figure(figsize=(10, 6))
    
    for algo in number_clusters.columns:
        plt.plot(param_range, number_clusters[algo].astype(float), marker='o', label=algo)

    if true_K is not None:
        plt.axhline(true_K, linestyle="--", color="black", label=f"True K = {true_K}")

    plt.xlabel(param_name)
    plt.ylabel("Inferred Number of Clusters")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_results_range_p(results_mean, results_std, modify: str="out", title: str=None):
    from IPython.display import clear_output
    clear_output(wait=True)
    
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

            axs[i].plot(range_p, mean_values, marker='x', label=algo)
            axs[i].fill_between(range_p, mean_values - std_values, mean_values + std_values, alpha=0.2)

        axs[i].set_title(metric.replace("_", " ").title())
        axs[i].set_xlabel("Probability p")
        axs[i].set_ylabel("Score")
        axs[i].legend()
        axs[i].grid(True)
        
    f.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

def plot_results_range_n(results_mean, results_std, modify: str="out", title: str=None):
    from IPython.display import clear_output
    clear_output(wait=True)
    
    algos = results_mean.columns 
    metrics = results_mean.index.get_level_values("Metric").unique()
    range_n = results_mean.index.get_level_values(f"n").unique()  

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

            axs[i].plot(range_n, mean_values, marker='x', label=algo)
            axs[i].fill_between(range_n, mean_values - std_values, mean_values + std_values, alpha=0.2)

        axs[i].set_title(metric.replace("_", " ").title())
        axs[i].set_xlabel("Number of nodes n")
        axs[i].set_ylabel("Score")
        axs[i].legend()
        axs[i].grid(True)
        
    f.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
