import time 
import pandas as pd 
from graph_generation import *
from validation import *
import matplotlib.pyplot as plt 

import os


algorithms = ["bayesian", "bayesian_fixed_K", "spectral", "leiden", "louvain"]
possible_metrics = [
    "adjusted_mutual_info_score", 
    "adjusted_rand_score", 
    "v_measure_score",
    "homogeneity_score",
    "fowlkes_mallows_score"
]


def compute_running_time(metric, algorithm, graphs, n_runs: int=10):
    """Compute running time for a certain (metric, algorithm) pair in order to compare algorithms' performance.

    Args:
        metric (str)
        algorithm (str)
        graphs (dict): Dictionary of ig.Graph to compute the scores on.
        n_runs (int, optional): Number of runs, in order to take the average. Defaults to 10.

    Returns:
        avg_time (float): Average time for the pair after n_runs.
    """
    times = []
    for _ in range(n_runs):
        start = time.time()
        _, _, _ = compute_score(metric, algorithm, graphs)
        times.append(time.time() - start)
    avg_time = np.mean(times)
    return avg_time


def running_time_analysis(K: int=3, nb_probas: int=5, modify : str="out", plot: bool=True):
    """Generate table with the average running times for each possible pairs (metric, algorithm).

    Args:
        K (int, optional): Number of clusters for the graph generation. Defaults to 3.
        nb_probas (int, optional): Number of graphs to generate. Defaults to 5.
        modify (str, optional): Probability to modify is our SBM graphs generation. Defaults to "out".
        plot (bool, optional): If true, plot the obtained result. Defaults to True.

    Returns:
        results (pd.DataFrame): Dataframe with the average runnin g time for each pair.
    """
    graphs, _ = sbm_generation(K=K, nb_probas=nb_probas, modify=modify)
    results = pd.DataFrame(index=possible_metrics, columns=algorithms)
    for algorithm in algorithms:
        for metric in possible_metrics:
            time = compute_running_time(metric, algorithm, graphs)
            results.loc[metric, algorithm] = time
    
    ## Save results in a csv file
    os.makedirs("time_evaluations", exist_ok=True)
    name_table = f"time_evaluations/time_K{K}_n{nb_probas}.csv"
    results.to_csv(name_table, index=True)
    
    if plot:
        plot_rt(name_table)
    
    return results

def plot_rt(csv_path: str=None):
    from IPython.display import clear_output
    clear_output(wait=True)
    
    df = pd.read_csv(csv_path, index_col=0)
    
    plt.figure(figsize=(10, 6))
    for algo in df.columns:
        plt.plot(df.index, df[algo], marker="x", label=algo)
    
    plt.title("Running Time per Algorithm and Metric")
    plt.ylabel("Time (seconds)")
    plt.xlabel("Metric")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def running_time_vs_n(range_n: np.array=None, K: int=3, n_runs: int=10, metric: str="adjusted_rand_score", plot: 
bool=True):
    """Compute the running time for graphs of varying size, i.e., different number of nodes, all other graph generation's arguments are fixed ."""
    if range_n is None or len(range_n)==0:
        range_n = np.linspace(100*K, 1000*K, 10, dtype=np.int32)
    print(range_n)
    
    results = pd.DataFrame(index=range_n, columns=algorithms)
    
    for n in range_n:
        sbm_graphs, _ = sbm_generation(n=n, K=K, nb_probas=1)
        for algorithm in algorithms:
            time = compute_running_time(metric, algorithm, sbm_graphs, n_runs=n_runs)
            results.loc[n, algorithm] = time
        print(f"{n} done!")
    
    ## Save result in a csv
    name_table = f"time_evaluations/runtime_vs_n_K{K}"
    results.to_csv(name_table)
    
    if plot:
        plot_runtime_vs_n(results)
    
    return results

def plot_runtime_vs_n(df, title="Runtime vs Number of Nodes"):
    from IPython.display import clear_output
    clear_output(wait=True)
    plt.figure(figsize=(10, 6))
    for algo in df.columns:
        plt.plot(df.index, df[algo].astype(float), marker="o", label=algo)
    plt.xlabel("Number of Nodes (n)")
    plt.ylabel("Average Runtime (seconds)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def running_time_vs_K(
    range_K=np.arange(3, 11), 
    n=1000, 
    metric="adjusted_rand_score", 
    nb_probas=1, 
    n_runs=10,
    plot: bool=True
):
    """Compute the running time for various number of clusters, all other graph generation's arguments are fixed."""
    os.makedirs("time_evaluations", exist_ok=True)
    
    results = pd.DataFrame(index=range_K, columns=algorithms)

    for K in range_K:
        print(f"\nTesting K = {K}")
        sbm_graphs, _ = sbm_generation(n=n, K=K, nb_probas=nb_probas)
        
        for algo in algorithms:
            t = compute_running_time(metric, algo, sbm_graphs, n_runs=n_runs)
            results.loc[K, algo] = t

    # Save results
    file_path = f"time_evaluations/runtime_vs_K_n{n}.csv"
    results.to_csv(file_path)
    print(f"\nSaved to {file_path}")
    
    if plot:
        plot_runtime_vs_K(results)
    return results


def plot_runtime_vs_K(df, title="Runtime vs Number of Communities"):
    from IPython.display import clear_output
    clear_output(wait=True)
    
    plt.figure(figsize=(10, 6))
    for algo in df.columns:
        plt.plot(df.index, df[algo].astype(float), marker="o", label=algo)
    plt.xlabel("Number of Communities (K)")
    plt.ylabel("Average Runtime (seconds)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

