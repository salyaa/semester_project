import numpy as np 
import pandas as pd 
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, v_measure_score, homogeneity_score, fowlkes_mallows_score
import igraph as ig
import networkx as nx
import graph_tool.all as gt
import matplotlib.pyplot as plt

from SBM.sbm_generation import sbm_generation
from ABCD.abcd_generation import *


algorithms = ["bayesian", "bayesian_fixed_K", "spectral", "leiden", "louvain"]
algo_fixed_K = ["bayesian_fixed_K", "spectral"]
algo_not_fixed_K = [algo for algo in algorithms if algo not in algo_fixed_K]
possible_metrics = [
    "adjusted_mutual_info_score", 
    "adjusted_rand_score", 
    "v_measure_score",
    "homogeneity_score",
    "fowlkes_mallows_score"
]
graph_types = ["sbm", "abcd"]


def compute_score(metric, algorithm, graphs, block_membership, graph_type: str="sbm"):
    """Compute a given index for a dictionary of graphs

    Args:
        index (function): the index/score to compute, e.g., adjusted_mutual_info_score
        graphs (dict()): dictionary of the graphs (ig.Graph) we want to study
        algorithm (function): algorithm we want to apply on each graphs

    Returns:
        score (dict()): return a dictionary of the wanted score for each graph
        avg_score (float): return the average score
        std_score (float): return the standard deviation
        number_clusters_found (dict()): number of clusters inferred by the algorithm, useful to compare it with the real number of clusters
    """
    assert algorithm in algorithms, "The algorithm must be one that we implemented."
    assert metric in possible_metrics, "The metric must be one that we selected."
    assert graph_type in graph_types, "The type of graph must be one that we implemented."
    
    metric_functions = {
        "adjusted_mutual_info_score": adjusted_mutual_info_score,
        "adjusted_rand_score": adjusted_rand_score,
        "v_measure_score": v_measure_score, #identical to normalized_mutual_info_score with the 'arithmetic' option for averaging.
        "homogeneity_score": homogeneity_score,
        "fowlkes_mallows_score": fowlkes_mallows_score
    }
    compute_metric = metric_functions[metric] 
    
    scores = dict()
    number_clusters_found = {}
    
    for key, graph in graphs.items():
        true_clusters = block_membership[key]
        
        #useful for graph conversion
        if isinstance(graph, gt.Graph):
            graph_edges = [(int(e.source()), int(e.target())) for e in graph.edges()]
        else: 
            graph_edges = graph.get_edgelist()

        if algorithm in ["bayesian", "bayesian_fixed_K"]:
            from algorithms.statistical import bayesianInf, bayesianInfFixedK
            if isinstance(graph, ig.Graph):
                    gt_graph = gt.Graph(directed=False)
                    gt_graph.add_vertex(n=graph.vcount())
                    gt_graph.add_edge_list(graph_edges)
                    graph = gt_graph
            if algorithm == "bayesian":
                clusters = bayesianInf(graph)
            else:
                clusters = bayesianInfFixedK(G=graph, K=len(set(true_clusters)))
            predicted_clusters = clusters.a 
            
        elif algorithm == "spectral":
            from algorithms.spectral import spectral
            graph_nx = nx.Graph()
            if isinstance(graph, ig.Graph):
                node_mapping = {v.index: i for i, v in enumerate(graph.vs)}
                graph_nx.add_nodes_from(node_mapping.values())
                graph_nx.add_edges_from([(node_mapping[u], node_mapping[v]) for u, v in graph.get_edgelist()])
            else:
                node_mapping = {v: i for i, v in enumerate(graph.vertices())}
                graph_nx.add_nodes_from(node_mapping.values())
                graph_nx.add_edges_from([(node_mapping[int(e.source())], node_mapping[int(e.target())]) for e in graph.edges()])
            if not nx.is_connected(graph_nx):
                lcc = max(nx.connected_components(graph_nx), key=len)
                graph_nx = graph_nx.subgraph(lcc).copy()
                true_clusters = np.array([true_clusters[node] for node in graph_nx.nodes()])
            predicted_clusters = spectral(graph_nx, K=len(set(true_clusters)))
        
        elif algorithm in ["louvain", "leiden"]: # leiden and louvain
            ## Need to convert gt.Graph into ig.Graph
            if isinstance(graph, gt.Graph):
                graph_ig = ig.Graph()
                graph_ig.add_vertices(n=graph.num_vertices())
                graph_ig.add_edges(graph_edges)
                graph = graph_ig
            
            from algorithms.modularity_based import louvain, leiden
            method = leiden if algorithm == "leiden" else louvain
            clusters = method(graph)
            predicted_clusters = clusters.membership
        
        # Had issue with graph conversion, so test this:
        if len(true_clusters) != len(predicted_clusters):
            raise ValueError(f"Size mismatch: True labels ({len(true_clusters)}) vs. Predicted labels ({len(predicted_clusters)}) in {key}")
        
        score = compute_metric(true_clusters, predicted_clusters)
        scores[key] = score
        number_clusters_found[key] = len(np.unique(predicted_clusters))

    avg_score = np.mean(list(scores.values()))
    std_score = np.std(list(scores.values()))
    
    return scores, avg_score, std_score, number_clusters_found

def validation_range_K(range_K, modify: str="out", plot: bool=True, graph_type="sbm"):
    """Compute the score for all possible pairs (index, algorithm) for various number of true communities, tested on a single graph.

    Args:
        range_K (np.array): The different values of clusters we consider.
        modify (str, optional): Decide if we modify p_in or p_out. Defaults to "out".
        plot (boolean, optional): if True, plot the results over range_K for each score. Defaults to True.
        graph_type (str): "sbm" or "abcd"

    Returns:
        results (pd.DataFrame): Dataframe with multiple indexes (K, index) representing the average score for each pair.
        number_clusters (pd.DataFrame): Dataframe with the number of clusters found for each algorithm for a given true K.
    """
    param_name = "K"

    multi_ind = pd.MultiIndex.from_product(
        [range_K, possible_metrics], names=[param_name, "Metric"]
    )
    results_mean = pd.DataFrame(index=multi_ind, columns=algorithms)
    results_std = pd.DataFrame(index=multi_ind, columns=algorithms)

    number_clusters = pd.DataFrame(index=range_K, columns=algorithms)

    for K in range_K:
        if graph_type == "sbm":
            graphs, _, memberships = sbm_generation(K=K, nb_probas=1, modify=modify)
        elif graph_type == "abcd":
            graphs, memberships = abcd_equal_size_range_K(range_K=[K])
        else:
            raise ValueError("Invalid graph type. Must be 'sbm' or 'abcd'.")

        for algorithm in algorithms:
            all_cluster_counts = {}

            for metric in possible_metrics:
                _, avg_score, std_score, cluster_dict = compute_score(
                    metric, algorithm, graphs, memberships, graph_type=graph_type
                )
                results_mean.loc[(K, metric), algorithm] = avg_score
                results_std.loc[(K, metric), algorithm] = std_score

                # Store number of clusters found for each graph
                for gname, count in cluster_dict.items():
                    if gname not in all_cluster_counts:
                        all_cluster_counts[gname] = []
                    all_cluster_counts[gname].append(count)

            # Compute average number of clusters per graph, then average across graphs
            avg_cluster_counts = [np.mean(counts) for counts in all_cluster_counts.values()]
            number_clusters.loc[K, algorithm] = np.mean(avg_cluster_counts)

        print(f"Completed {param_name} = {K}.")

    ## Save the results in a csv file
    name_table = f"evaluations/scores_{param_name}{range_K[0]}to{range_K[-1]}_{graph_type}_p{modify}.csv"
    results_mean.to_csv(name_table)
    print(f"Results saved at {name_table}!")

    if plot:
        title = f"Community detection performance across different scores ({graph_type.upper()}), varying the number of clusters K"
        #plot_results_range_K(results_mean, results_std, title, number_clusters)
        plot_results_generic(results_mean, results_std, param_name, "K", title)
        plot_inferred_K(number_clusters, range_K, param_name=param_name)

    return results_mean, number_clusters

def validation_range_p(range_p: np.array=None, K: int=3, n: int=1000, modify: str="out", graph_type: str="sbm", plot: bool=True):
    """Compute the average score for all possible pairs (index, algorithm) for a fixed number of true communities K, tested for various SBM or ABCD graphs modifying p_in or p_out depending on `modify`.

    Args:
        range_p (np.array, optional): The different probability values we want to test out. Defaults to None.
        K (int, optional): Number of true communities to consider. Defaults to 3.
        n (int, optional): Number of nodes. Used for ABCD only.
        modify (str, optional): Define if we modify p_in or p_out. Defaults to "out".
        graph_type (str): "sbm" or "abcd"
        plot (bool, optional): If True, plot the results over range_p for each score. Defaults to True.

    Returns:
        results (pd.DataFrame): Dataframe with multiple indexes (p, metric) representing the average score for each pair.
        number_clusters (pd.DataFrame): Dataframe with the number of clusters found for each algorithm and p value.
    """
    if graph_type == "sbm":
        param_name = f"p_{modify}"
    elif graph_type == "abcd":
        param_name = "xi"

    if range_p is None:
        if graph_type == "sbm": # entre 0 et 4*K*log(n)/n
            range_p = np.linspace(0, 4*K*np.log(n)/n, 11) 
        else:
            range_p = np.round(np.linspace(0.1, 0.9, 9), 2)

    multi_ind = pd.MultiIndex.from_product(
        [range_p, possible_metrics], names=[param_name, "Metric"]
    )
    results = pd.DataFrame(index=multi_ind, columns=algorithms)
    results_std = pd.DataFrame(index=multi_ind, columns=algorithms)
    number_clusters = pd.DataFrame(index=range_p, columns=algorithms)

    for p in range_p:
        if graph_type == "sbm":
            graphs, _, memberships = sbm_generation(K=K, range_p=np.array([p]), modify=modify)
        elif graph_type == "abcd":
            graphs, _, memberships = abcd_equal_size_range_xi(range_xi=[p], K=K, n=n)
        else:
            raise ValueError("Invalid graph type. Must be 'sbm' or 'abcd'.")

        for algorithm in algorithms:
            k_algo_counts = []
            for metric in possible_metrics:
                _, avg_score, std_score, cluster_dict = compute_score(
                    metric, algorithm, graphs, memberships, graph_type=graph_type
                )
                results.loc[(p, metric), algorithm] = avg_score
                results_std.loc[(p, metric), algorithm] = std_score
                k_algo_counts.extend(cluster_dict.values())  # Flatten dict to list of ints
            number_clusters.loc[p, algorithm] = np.mean(k_algo_counts)

        print(f"Completed {param_name} = {p}")

    # Save results
    os.makedirs("evaluations", exist_ok=True)
    name_table = f"evaluations/scores_K{K}_n{len(range_p)}_{graph_type}_{param_name}.csv"
    results.to_csv(name_table)
    print(f"Results saved at {name_table}!")

    if plot:
        if graph_type == "sbm":
            title = f"Community detection average performance for {K} communities across different scores (using SBM graphs), modifying {param_name}"
        elif graph_type == "abcd":
            title = f"Community detection average performance for {K} communities across different scores (using ABCD graphs), modifying xi"
        plot_results_generic(results, results_std, param_name, title=title)
        plot_inferred_K_fixed_param(number_clusters, range_p, param_name, K, f"Inferred Clusters vs {param_name} (K={K})")

    return results, number_clusters

def validation_range_n(range_n: np.array=None, K: int=3, modify: str="out", graph_type: str="sbm", plot: bool=True):
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
    param_name = "n"
    
    if range_n is None:
        range_n = np.linspace(100*K, 1000*K, 10, dtype=np.int32)
        if graph_type=="abcd":
            range_n = [n for n in range_n if n%K == 0]
    print(range_n)
    
    multi_ind = pd.MultiIndex.from_product(
        [range_n, possible_metrics], names=[param_name, "Metric"]
    )
    results = pd.DataFrame(index=multi_ind, columns=algorithms)
    results_std = pd.DataFrame(index=multi_ind, columns=algorithms)
    
    number_clusters = pd.DataFrame(index=range_n, columns=algorithms)
    
    for n in range_n:
        assert n%K == 0
        if graph_type == "sbm":
            graphs, _, memberships = sbm_generation(n=n, K=K, nb_probas=5, modify=modify)
        elif graph_type=="abcd":
            graphs, _, memberships = abcd_equal_size_range_xi(num_graphs=5, xi_max=0.4, n=n, K=K)
        for algorithm in algorithms:
            k_algo_counts = []
            for metric in possible_metrics:
                _, avg_score, std_score, cluster_dict = compute_score(
                    metric, algorithm, graphs, memberships, graph_type=graph_type
                )
                results.loc[(n, metric), algorithm] = avg_score
                results_std.loc[(n, metric), algorithm] = std_score
                k_algo_counts.extend(cluster_dict.values())
            number_clusters.loc[n, algorithm] = np.mean(k_algo_counts)
        print(f"Completed {param_name} = {n}.")
        
    ## Save the results in a csv file
    name_table = f"evaluations/scores_K{K}_{param_name}_from{range_n[0]}to{range_n[-1]}_p{modify}.csv"
    results.to_csv(name_table)
    print(f"Results saved at {name_table}!")
    
    if plot:
        title = f"Community detection average performance for {K} communities across different scores, using {graph_type.upper()} graphs varying the number of nodes n"
        plot_results_generic(results, results_std, param_name, title=title)
        plot_inferred_K_fixed_param(
            number_clusters, 
            param_range=range_n, 
            param_name=param_name, 
            true_K=K,
            title=f"Inferred Clusters vs Network Size n (K={K})"
        )
    
    return results, number_clusters

## 2 more validation functions for ABCD graphs (for c_min and d_max):
def validation_range_c_min(num_graphs: int, K: int=5,  n: int=1000, c_max: int=1000, xi_max: float=0.5, plot: bool=True):
    param_name = "c_min"
    
    assert n%K == 0, f"n={n} must be divisible by K={K} for equal-sized communities."
    max_val = int(n/K)
    assert max_val <= c_max
    range_c = np.linspace(20, max_val, num_graphs, dtype=int)
    
    multi_ind = pd.MultiIndex.from_product(
        [range_c, possible_metrics], names=[param_name, "Metric"]
    )
    results = pd.DataFrame(index=multi_ind, columns=algorithms)
    results_std = pd.DataFrame(index=multi_ind, columns=algorithms)
    number_clusters = pd.DataFrame(index=range_c, columns=algorithms)
    
    for c in range_c:
        graphs, _, memberships = abcd_equal_size_range_xi(num_graphs=5, xi_max=xi_max, n=n, K=K, c_min=c, c_max=c_max)
        for algorithm in algorithms:
            k_algo_counts = []
            for metric in possible_metrics:
                _, avg_score, std_score, cluster_dict = compute_score(
                    metric, algorithm, graphs, memberships, graph_type="abcd"
                )
                results.loc[(c, metric), algorithm] = avg_score
                results_std.loc[(c, metric), algorithm] = std_score
                k_algo_counts.extend(cluster_dict.values())
            number_clusters.loc[c, algorithm] = np.mean(k_algo_counts)
        print(f"Completed {param_name} = {c}.")
        
    ## Save the results in a csv file
    name_table = f"evaluations/scores_K{K}_{param_name}_from{range_c[0]}to{range_c[-1]}_ABCD.csv"
    results.to_csv(name_table)
    print(f"Results saved at {name_table}!")
    
    if plot:
        title = f"Community detection average performance for {K} communities across different scores, using ABCD graphs varying {param_name}"
        plot_results_generic(results, results_std, param_name, title=title)
        plot_inferred_K_fixed_param(
            number_clusters, 
            param_range=range_c, 
            param_name=param_name, 
            true_K=K,
            title=f"Inferred Clusters vs c_min (K={K})"
        )
    
    return results, number_clusters

def validation_range_d_max(num_graphs: int, K: int=5,  n: int=1000, d_min: int=5, xi_max:float=0.5, plot: bool=True):
    assert n%K == 0, f"n={n} must be divisible by K={K} for equal-sized communities."
    max_val = n//10
    assert max_val >= d_min
    range_d = np.linspace(20, max_val, num_graphs, dtype=int)
    
    param_name = "d_max"
    
    multi_ind = pd.MultiIndex.from_product(
        [range_d, possible_metrics], names=[param_name, "Metric"]
    )
    results = pd.DataFrame(index=multi_ind, columns=algorithms)
    results_std = pd.DataFrame(index=multi_ind, columns=algorithms)
    number_clusters = pd.DataFrame(index=range_d, columns=algorithms)
    
    for d in range_d:
        graphs, _, memberships = abcd_equal_size_range_xi(num_graphs=5, xi_max=xi_max, n=n, K=K, d_min=d_min, d_max=d)
        for algorithm in algorithms:
            k_algo_counts = []
            for metric in possible_metrics:
                _, avg_score, std_score, cluster_dict = compute_score(
                    metric, algorithm, graphs, memberships, graph_type="abcd"
                )
                results.loc[(d, metric), algorithm] = avg_score
                results_std.loc[(d, metric), algorithm] = std_score
                k_algo_counts.extend(cluster_dict.values())
            number_clusters.loc[d, algorithm] = np.mean(k_algo_counts)
        print(f"Completed {param_name} = {d}.")
        
    ## Save the results in a csv file
    name_table = f"evaluations/scores_K{K}_dmax_from{range_d[0]}to{range_d[-1]}_ABCD.csv"
    results.to_csv(name_table)
    print(f"Results saved at {name_table}!")
    
    if plot:
        title = f"Community detection average performance for {K} communities across different scores, using ABCD graphs varying d_max"
        plot_results_generic(results, results_std, param_name=param_name, title=title)
        plot_inferred_K_fixed_param(
            number_clusters, 
            param_range=range_d, 
            param_name=param_name, 
            true_K=K,
            title=f"Inferred Clusters vs d_max (K={K})"
        )
    
    return results, number_clusters

### PLOT FUNCTIONS ###
def plot_inferred_K(number_clusters, true_K_range, param_name: str, title="Inferred Number of Clusters"):
    """ Plot the number of clusters found by each algorithm for a given true K.
    The true K is represented by the dashed line.

    Args:
        number_clusters (dict()): Dataframe with the number of clusters found for each algorithm for a given true K.
        true_K_range (np.array): The range of true K values.
        title (str, optional): Title of the plot. Defaults to "Inferred Number of Clusters".
    """
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

def plot_inferred_K_fixed_param(number_clusters, param_range, param_name: str, true_K=None, title="Inferred Number of Clusters"):
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

def plot_results_generic(
    results_mean,
    results_std,
    param_name: str,
    param_label: str = None,
    title: str = None
):
    """
    Generic plotting function for evaluation results over a varying parameter.
    Shows two rows: one for fixed-K algorithms, one for non-fixed-K algorithms.

    Args:
        results_mean (pd.DataFrame): Mean scores, indexed by MultiIndex (param_value, "Metric")
        results_std (pd.DataFrame): Standard deviations, same shape as results_mean
        param_name (str): The name of the parameter that was varied (e.g., "K", "n", "p_out", "d", "c")
        param_label (str): Label for the x-axis (optional). If None, defaults to param_name.
        title (str): Title of the full figure.
    """
    from IPython.display import clear_output
    clear_output(wait=True)

    all_algos = results_mean.columns
    metrics = results_mean.index.get_level_values("Metric").unique()
    param_values = results_mean.index.get_level_values(param_name).unique()

    fixed_K_algos = ["bayesian_fixed_K", "spectral"]
    non_fixed_K_algos = [a for a in all_algos if a not in fixed_K_algos]

    f, axs = plt.subplots(2, len(metrics), figsize=(5 * len(metrics), 10), sharex=True, sharey=True)

    if len(metrics) == 1:
        axs = np.array([[axs[0]], [axs[1]]])  # Ensure 2D shape even with one column

    for col, metric in enumerate(metrics):
        avg_scores = results_mean.xs(metric, level="Metric")
        std_scores = results_std.xs(metric, level="Metric")

        for row, algo_group in enumerate([non_fixed_K_algos, fixed_K_algos]):
            ax = axs[row][col]
            for algo in algo_group:
                mean_values = avg_scores[algo].astype(float)
                std_values = std_scores[algo].astype(float)
                std_values = np.nan_to_num(std_values, nan=0.0)

                ax.plot(param_values, mean_values, marker='x', label=algo)
                ax.fill_between(param_values, mean_values - std_values, mean_values + std_values, alpha=0.2)

            ax.set_title(f"{metric.replace('_', ' ').title()} ({'Fixed K' if row == 1 else 'Inferred K'})")
            if row == 1:
                ax.set_xlabel(param_label if param_label else param_name)
            if col == 0:
                ax.set_ylabel("Score")
            ax.grid(True)
            ax.legend()

    f.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

