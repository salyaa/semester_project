import numpy as np 
import pandas as pd 
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, v_measure_score, homogeneity_score, fowlkes_mallows_score
import igraph as ig
import networkx as nx
import graph_tool.all as gt
import matplotlib.pyplot as plt
import matplotlib as mpl

from helpers import *
from plots import *
from constants import possible_metrics, algorithms, algo_fixed_K, algo_not_fixed_K

from SBM.sbm_generation import sbm_generation
from ABCD.abcd_generation import *
from helpers import *

def compute_score(metric, algorithm, graphs, block_membership, possible_algos=algorithms, graph_type: str="sbm"):
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
    assert metric in possible_metrics, "The metric must be one that we selected."
    
    metric_functions = {
        "adjusted_mutual_info_score": adjusted_mutual_info_score,
        "adjusted_rand_score": adjusted_rand_score,
        "v_measure_score": v_measure_score,
        "homogeneity_score": homogeneity_score,
        "fowlkes_mallows_score": fowlkes_mallows_score
    }
    compute_metric = metric_functions[metric] 
    
    scores = dict()
    memberships, true_clusters = get_predicted_memberships(algorithm, graphs, block_membership, possible_algos=possible_algos, graph_type=graph_type)
    for key, _ in graphs.items():
        scores[key] = compute_metric(true_clusters[key], memberships[key])
    number_clusters_found = {key: len(np.unique(memberships[key])) for key in memberships.keys()}
    
    avg_score = np.mean(list(scores.values()))
    std_score = np.std(list(scores.values()))
    
    return scores, avg_score, std_score, number_clusters_found

def validation_range_K(range_K, n: int=3000, n_graphs: int=15, modify: str="out", plot: bool=True, graph_type="sbm", possible_algorithms: list=algorithms):
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
    results_mean = pd.DataFrame(index=multi_ind, columns=possible_algorithms, dtype=float)
    results_std  = pd.DataFrame(index=multi_ind, columns=possible_algorithms, dtype=float)
    results_sem  = pd.DataFrame(index=multi_ind, columns=possible_algorithms, dtype=float)

    number_clusters = pd.DataFrame(index=range_K, columns=possible_algorithms)
    
    for K in range_K:
        print(f"Starting K = {K}...")
        if graph_type == "sbm":
            p = np.log(n)/n
            n_reps_p = np.full(n_graphs, p)
            graphs, _, memberships = sbm_generation(n=n, K=K, range_p=n_reps_p, modify=modify)
        elif graph_type == "abcd":
            graphs, memberships = abcd_equal_size_range_K(range_K=[K], n=n, n_graphs=n_graphs)
        else:
            raise ValueError("Invalid graph type. Must be 'sbm' or 'abcd'.")

        df = evaluate_all(graphs, memberships, possible_algos=possible_algorithms, graph_type=graph_type)
        means = df["mean"].unstack(0)
        sem = (df["std"]/np.sqrt(n_graphs)).unstack(0)
        means_ns = df["mean_n"].unstack(0).mean(axis=0)
        
        for algorithm in possible_algorithms:
            for metric in possible_metrics:
                #_, avg_score, std_score, cluster_dict = compute_score(
                #     metric, algorithm, graphs, memberships, possible_algos=possible_algorithms, graph_type=graph_type
                # )
                results_mean.loc[(K, metric), algorithm] = means.loc[metric, algorithm]
                results_sem.loc[(K, metric), algorithm]  = sem.loc[metric, algorithm]

                # Store number of clusters found for each graph
                # for gname, count in cluster_dict.items():
                #     if gname not in all_cluster_counts:
                #         all_cluster_counts[gname] = []
                #     all_cluster_counts[gname].append(count)
            number_clusters.loc[K, algorithm] = means_ns[algorithm]
            # Compute average number of clusters per graph, then average across graphs
            #avg_cluster_counts = [np.mean(counts) for counts in all_cluster_counts.values()]
            #number_clusters.loc[K, algorithm] = np.mean(avg_cluster_counts)

        print(f"Completed {param_name} = {K}.")

    ## Save the results in a csv file
    name_table = f"evaluations/scores_{param_name}{range_K[0]}to{range_K[-1]}_{graph_type}_p{modify}.csv"
    results_mean.to_csv(name_table)
    print(f"Results saved at {name_table}!")

    if plot:
        title = f"Community detection performance across different scores ({graph_type.upper()}), varying the number of clusters K (n={n})"
        #plot_results_range_K(results_mean, results_std, title, number_clusters)
        if possible_algorithms==algorithms_sc:
            plot_results_generic_sc(results_mean, results_sem, param_name, title=title)
        else:
            plot_results_generic(results_mean, results_sem, param_name, "K", title)
            #plot_single_score(results_mean, results_std, param_name, param_label="K", title=title)
            plot_inferred_K(number_clusters, range_K, param_name=param_name)

    return results_mean, number_clusters

def validation_range_p(range_p: np.array=None, K: int=3, n: int=3000, n_reps: int=15, modify: str="out", graph_type: str="sbm", possible_algorithms: list=algorithms, plot: bool=True):
    """Compute the average score for all possible pairs (index, algorithm) for a fixed number of true communities K, tested for various SBM or ABCD graphs modifying p_in or p_out depending on `modify`.

    Args:
        range_p (np.array, optional): The different probability values we want to test out. Defaults to None.
        K (int, optional): Number of true communities to consider. Defaults to 3.
        n (int, optional): Number of nodes. Used for ABCD only.
        n_reps (int, optional): Number of repetitions for each p value. Defaults to 15.
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
    results = pd.DataFrame(index=multi_ind, columns=possible_algorithms)
    results_sem = pd.DataFrame(index=multi_ind, columns=possible_algorithms)
    number_clusters = pd.DataFrame(index=range_p, columns=possible_algorithms)

    for p in range_p:
        if graph_type == "sbm":
            n_reps_p = np.full(n_reps, p)
            graphs, _, memberships = sbm_generation(n=n, K=K, range_p=n_reps_p, modify=modify)
        elif graph_type == "abcd":
            graphs, _, memberships = abcd_equal_size_range_xi(range_xi=[p], n_reps=n_reps, K=K, n=n)
        else:
            raise ValueError("Invalid graph type. Must be 'sbm' or 'abcd'.")
        df = evaluate_all(graphs, memberships, possible_algos=possible_algorithms, graph_type=graph_type)
        means = df["mean"].unstack(0)
        sem = (df["std"]/np.sqrt(n_reps)).unstack(0)
        means_ns = df["mean_n"].unstack(0).mean(axis=0)
        for algorithm in possible_algorithms:
            for metric in possible_metrics:
                results.loc[(p, metric), algorithm] = means.loc[metric, algorithm]
                results_sem.loc[(p, metric), algorithm]  = sem.loc[metric, algorithm]
                # _, avg_score, std_score, cluster_dict = compute_score(
                #     metric, algorithm, graphs, memberships,possible_algos=possible_algorithms, graph_type=graph_type
                # )
                # results.loc[(p, metric), algorithm] = avg_score
                # results_std.loc[(p, metric), algorithm] = std_score
                # results_sem.loc[(p, metric), algorithm] = std_score / np.sqrt(actual_rep)
                #k_algo_counts.extend(cluster_dict.values())  # Flatten dict to list of ints
            #number_clusters.loc[p, algorithm] = np.mean(k_algo_counts)
            number_clusters.loc[p, algorithm] = means_ns[algorithm]

        print(f"Completed {param_name} = {p}")

    # Save results
    os.makedirs("evaluations", exist_ok=True)
    name_table = f"evaluations/scores_K{K}_n{len(range_p)}_{graph_type}_{param_name}.csv"
    results.to_csv(name_table)
    print(f"Results saved at {name_table}!")

    if plot:
        if graph_type == "sbm":
            title = f"Community detection average performance for {K} communities across different scores (using SBM graphs), modifying {param_name} (n={n})"
        elif graph_type == "abcd":
            title = f"Community detection average performance for {K} communities across different scores (using ABCD graphs), modifying xi"
        
        if possible_algorithms==algorithms_sc:
            plot_results_generic_sc(results, results_sem, param_name, title=title)
        else:
            plot_results_generic(results, results_sem, param_name, title=title)
            #plot_single_score(results, results_std, param_name, title=title)
            plot_inferred_K_fixed_param(number_clusters, range_p, param_name, K, f"Inferred Clusters vs {param_name} (K={K}, n={n})")

    return results, number_clusters

def validation_range_n(range_n: np.array=None, K: int=10, n_graphs: int=5, modify: str="out", graph_type: str="sbm", possible_algorithms: list=algorithms, plot: bool=True):
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
    results = pd.DataFrame(index=multi_ind, columns=possible_algorithms)
    results_sem = pd.DataFrame(index=multi_ind, columns=possible_algorithms)
    
    number_clusters = pd.DataFrame(index=range_n, columns=possible_algorithms)
    
    for n in range_n:
        if graph_type == "sbm":
            p = 2*K*np.log(n)/n
            n_reps_p = np.full(n_graphs, p/6)
            graphs, _, memberships = sbm_generation(n=n, K=K, range_p=n_reps_p, modify=modify)
        elif graph_type=="abcd":
            assert n%K == 0
            xi = 0.5  # fixed xi for ABCD graphs
            n_reps_xi = np.full(n_graphs, xi)
            graphs, _, memberships = abcd_equal_size_range_xi(range_xi=n_reps_xi,num_graphs=n_graphs, n=n, K=K)
        df = evaluate_all(graphs, memberships, possible_algos=possible_algorithms, graph_type=graph_type)
        means = df["mean"].unstack(0)
        sem = (df["std"]/np.sqrt(n_graphs)).unstack(0)
        means_ns = df["mean_n"].unstack(0).mean(axis=0)
        for algorithm in possible_algorithms:
            k_algo_counts = []
            for metric in possible_metrics:
                results.loc[(n, metric), algorithm] = means.loc[metric, algorithm]
                results_sem.loc[(n, metric), algorithm]  = sem.loc[metric, algorithm]
            number_clusters.loc[n, algorithm] = means_ns[algorithm]
        print(f"Completed {param_name} = {n}.")
    
    ## Save the results in a csv file
    name_table = f"evaluations/scores_K{K}_{param_name}_from{range_n[0]}to{range_n[-1]}_p{modify}.csv"
    results.to_csv(name_table)
    print(f"Results saved at {name_table}!")
    
    if plot:
        title = f"Community detection average performance for {K} communities across different scores, using {graph_type.upper()} graphs varying the number of nodes n"
        if possible_algorithms==algorithms_sc:
            plot_results_generic_sc(results, results_sem, param_name, title=title)
        else:
            plot_results_generic(results, results_sem, param_name, title=title)
            #plot_single_score(results, results_std, param_name, title=title)
            plot_inferred_K_fixed_param(
                number_clusters, 
                param_range=range_n, 
                param_name=param_name, 
                true_K=K,
                title=f"Inferred Clusters vs Network Size n (K={K})"
            )
    
    return results, number_clusters

def validation_range_c_min(
    num_graphs: int,
    K: int = 15,
    n: int = 5000,
    c_max: int = 1000,
    xi_max: float = 0.5,
    plot: bool = True
):
    """
    Evaluate performance across different c_min values for ABCD graphs.

    Returns:
      results (DataFrame): mean scores indexed by (c_min, metric) × algorithm
      number_clusters (DataFrame): mean number of clusters found per c_min × algorithm
    """
    param_name = "c_min"
    assert n % K == 0, "n must be divisible by K"
    max_c = n // K
    assert max_c <= c_max

    # choose c_min values
    range_c = np.linspace(20, max_c, num_graphs, dtype=int)

    # prepare results DataFrames
    mindex         = pd.MultiIndex.from_product([range_c, possible_metrics],
                                                names=[param_name, "Metric"])
    results        = pd.DataFrame(index=mindex, columns=algorithms, dtype=float)
    results_std    = pd.DataFrame(index=mindex, columns=algorithms, dtype=float)
    number_clusters = pd.DataFrame(index=range_c, columns=algorithms, dtype=float)

    # containers for cluster sizes
    cluster_sizes      = {c: {} for c in range_c}
    true_cluster_sizes = {}

    for c in range_c:
        print(f"→ Generating ABCD graphs for c_min={c}")
        # generate graphs
        graphs, _, true_members = abcd_equal_size_range_xi(
            range_xi=None,
            num_graphs=num_graphs,
            n_reps=1,
            xi_max=xi_max,
            n=n,
            K=K,
            c_min=c,
            c_max=c_max
        )

        # true cluster-size distribution
        ts = []
        for b in true_members.values():
            b = np.array(b, dtype=int)
            # if your memberships are 1…K, shift them to 0…K-1:
            if b.min() == 1 and b.max() == K:
                b = b - 1
            _, counts = np.unique(b, return_counts=True)
            ts.extend(counts.tolist())
        true_cluster_sizes[c] = ts

        # batch-compute metrics
        df    = evaluate_all(graphs, true_members,
                            possible_algos=algorithms,
                            graph_type="abcd")
        means = df["mean"].unstack(level=1)
        stds  = df["std" ].unstack(level=1)
        

        # fill results for this c
        for metric in possible_metrics:
            for algo in algorithms:
                results.loc[(c, metric), algo]     = means.loc[(algo, metric)]
                results_std.loc[(c, metric), algo] = stds.loc[(algo, metric)]

        # mean number of clusters found
        mean_nfound = df["mean_n"].unstack(level=1).mean(axis=1)
        for algo in algorithms:
            number_clusters.loc[c, algo] = mean_nfound.loc[algo]

        # predicted cluster-size distributions
        for algo in algorithms:
            preds, _ = get_predicted_memberships(
                algo, graphs, true_members,
                possible_algos=algorithms,
                graph_type="abcd"
            )
            flat = np.concatenate([np.bincount(p) for p in preds.values()])
            cluster_sizes[c][algo] = flat.tolist()

        print(f"✓ Done c_min={c}")

    # save mean scores
    out_path = f"evaluations/scores_K{K}_{param_name}_from{range_c[0]}to{range_c[-1]}_ABCD.csv"
    results.to_csv(out_path)
    print(f"Saved results to {out_path}")

    if plot:
        title = f"ABCD performance vs {param_name} (K={K}, n={n}, xi_max={xi_max})"
        plot_results_generic(results, results_std, param_name, title=title)

        plot_inferred_K(
            number_clusters,
            param_range=range_c,
            param_name=param_name,
            true_K=K,
            title=f"Inferred Clusters vs {param_name}"
        )

        plot_cluster_size_boxplots(
            cluster_sizes,
            true_cluster_sizes,
            param_range=range_c,
            param_name=param_name,
            title=f"Cluster-size distributions vs {param_name}"
        )

    return results, number_clusters

def validation_range_d_max(
    num_graphs: int,
    K: int = 5,
    n: int = 5000,
    d_min: int = 5,
    xi_max: float = 0.5,
    plot: bool = True
):
    assert n % K == 0, f"n={n} must be divisible by K={K}"
    param_name = "d_max"
    max_d = n // 10
    assert max_d >= d_min

    # 1) choose d_max values
    range_d = np.linspace(d_min, max_d, num_graphs, dtype=int)

    # 2) prepare result DataFrames
    mindex       = pd.MultiIndex.from_product([range_d, possible_metrics],
                                              names=[param_name, "Metric"])
    results      = pd.DataFrame(index=mindex, columns=algorithms, dtype=float)
    results_std  = pd.DataFrame(index=mindex, columns=algorithms, dtype=float)
    number_clusters = pd.DataFrame(index=range_d, columns=algorithms, dtype=float)

    # for boxplots
    cluster_sizes      = {d: {} for d in range_d}
    true_cluster_sizes = {}

    for d in range_d:
        print(f"→ Generating ABCD graphs for d_max = {d}")
        # generate graphs
        graphs, _, true_members = abcd_equal_size_range_xi(
            range_xi=None,
            num_graphs=num_graphs,
            n_reps=1,
            xi_max=xi_max,
            n=n,
            K=K,
            c_min=None,
            c_max=None,
            d_min=d_min,
            d_max=d
        )

        # build true‐size list (one count per real cluster per graph)
        ts = []
        for b in true_members.values():
            b = np.array(b, dtype=int)
            # if labels are 1..K shift to 0..K-1
            if b.min() == 1 and b.max() == K:
                b = b - 1
            _, counts = np.unique(b, return_counts=True)
            ts.extend(counts.tolist())
        true_cluster_sizes[d] = ts

        # ——— batch‐compute all means & stds ———
        df    = evaluate_all(graphs, true_members,
                             possible_algos=algorithms,
                             graph_type="abcd")
        # fill in results per (d, metric, algo)
        for algo in algorithms:
            for metric in possible_metrics:
                results.loc[(d, metric), algo]     = df.loc[(algo, metric), "mean"]
                results_std.loc[(d, metric), algo] = df.loc[(algo, metric), "std"]

        # ——— average number of clusters found ———
        mean_nfound = df["mean_n"].unstack(level=0).mean(axis=0)
        for algo in algorithms:
            number_clusters.loc[d, algo] = mean_nfound[algo]

        # ——— predicted cluster‐sizes for boxplots ———
        for algo in algorithms:
            preds, _ = get_predicted_memberships(
                algo, graphs, true_members,
                possible_algos=algorithms,
                graph_type="abcd"
            )
            flat = []
            for p in preds.values():
                p = np.array(p, dtype=int)
                if p.min() == 1 and p.max() == K:
                    p = p - 1
                flat.extend(np.bincount(p).tolist())
            cluster_sizes[d][algo] = flat

        print(f"✓ Done d_max = {d}")

    # 3) save means
    out_path = f"evaluations/scores_K{K}_{param_name}_from{range_d[0]}to{range_d[-1]}_ABCD.csv"
    results.to_csv(out_path)
    print(f"Results saved to {out_path}")

    # 4) optional plotting
    if plot:
        title = f"ABCD performance vs {param_name} (K={K}, n={n}, xi_max={xi_max})"
        plot_results_generic(results, results_std, param_name, title=title)

        plot_inferred_K(
            number_clusters,
            param_range=range_d,
            param_name=param_name,
            true_K=K,
            title=f"Inferred Clusters vs {param_name}"
        )

        plot_cluster_size_boxplots(
            cluster_sizes,
            true_cluster_sizes,
            param_range=range_d,
            param_name=param_name,
            title=f"Cluster‐size distributions vs {param_name}"
        )

    return results, number_clusters
