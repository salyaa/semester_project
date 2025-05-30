import time 
import pandas as pd 
import numpy as np 
from SBM.sbm_generation import *
from ABCD.abcd_generation import *
from validation import *
import matplotlib.pyplot as plt 

import os

from constants import *


def compute_running_time(metric, algorithm, graphs, memberships, graph_type: str="sbm", n_runs: int=10, possible_algorithms=algorithms):
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
        start = time.perf_counter()
        _, _, _, _ = compute_score(metric, algorithm, graphs, memberships, possible_algorithms, graph_type)
        times.append(time.perf_counter() - start)
    avg_time = np.mean(times)
    return avg_time

def running_time_analysis(K: int=3, n_runs:int=5, nb_probas: int=5, modify : str="out", graph_type: str="sbm", possible_algorithms=algorithms, plot: bool=True):
    """Generate table with the average running times for each possible pairs (metric, algorithm).
    Args:
        K (int, optional): Number of clusters for the graph generation. Defaults to 3.
        nb_probas (int, optional): Number of graphs to generate. Defaults to 5.
        modify (str, optional): Probability to modify is our SBM graphs generation. Defaults to "out".
        plot (bool, optional): If true, plot the obtained result. Defaults to True.

    Returns:
        results (pd.DataFrame): Dataframe with the average runnin g time for each pair.
    """
    assert graph_type in graph_types
    
    if graph_type == "sbm":
        graphs, _, b = sbm_generation(K=K, nb_probas=nb_probas, modify=modify)
    elif graph_type == "abcd":
        graphs, _, b = abcd_equal_size_range_xi(num_graphs=nb_probas, K=K)
    results = pd.DataFrame(index=possible_metrics, columns=algorithms)
    for algorithm in possible_algorithms:
        for metric in possible_metrics:
            t = compute_running_time(metric, algorithm, graphs, b, graph_type, n_runs=n_runs, possible_algorithms=possible_algorithms)
            results.loc[metric, algorithm] = t
    
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

def running_time_vs_n(range_n: np.array=None, K: int=5, n_runs: int=10, metric: str="adjusted_rand_score", graph_type: str="sbm", possible_algorithms=algorithms, plot: bool=True):
    """Compute the running time for graphs of varying size, i.e., different number of nodes, all other graph generation's arguments are fixed ."""
    assert graph_type in graph_types
    
    if range_n is None or len(range_n)==0:
        range_n = np.linspace(100*K, 1000*K, 10, dtype=np.int32)
        range_n = [n for n in range_n if n%K == 0]
    print(range_n)
    
    results = pd.DataFrame(index=range_n, columns=possible_algorithms)
    
    for n in range_n:
        if graph_type == "sbm":
            graphs, _, b = sbm_generation(n=n, K=K, nb_probas=1)
        elif graph_type == "abcd":
            graphs, _, b = abcd_equal_size_range_xi(num_graphs=1, n=n, K=K)
        for algorithm in possible_algorithms:
            time = compute_running_time(metric, algorithm, graphs, b, graph_type, n_runs=n_runs, possible_algorithms=possible_algorithms)
            results.loc[n, algorithm] = time
        print(f"{n} done!")
    
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
    graph_type: str="sbm",
    possible_algorithms=algorithms,
    plot: bool=True
):
    """Compute the running time for various number of clusters, all other graph generation's arguments are fixed."""
    assert graph_type in graph_types
    assert metric in possible_metrics
    
    os.makedirs("time_evaluations", exist_ok=True)
    
    results = pd.DataFrame(index=range_K, columns=possible_algorithms)

    for K in range_K:
        print(f"\nTesting K = {K}")
        if graph_type == "sbm":
            graphs, _, b = sbm_generation(n=n, K=K, nb_probas=nb_probas)
        elif graph_type == "abcd":
            if n%K==0:
                graphs, _, b = abcd_equal_size_range_xi(num_graphs=nb_probas, n=n, K=K)
        
        for algo in possible_algorithms:
            t = compute_running_time(metric, algo, graphs, b, graph_type, n_runs, possible_algorithms)
            results.loc[K, algo] = t
    
    if plot:
        plot_runtime_vs_K(results)
    return results

def rt_vs_K_algos(range_K: np.array, n:int=3000, nb_probas: int=5, n_runs: int=10, graph_type: str="sbm", possible_algorithms = algorithms, plot: bool=True):
    assert graph_type in graph_types
    results = {}
    for metric in possible_metrics:
        print(f"\nTesting metric = {metric}")
        results[metric] = pd.DataFrame(index=range_K, columns=possible_algorithms)
        for K in range_K:
            if graph_type == "sbm":
                graphs, _, b = sbm_generation(n=n, K=K, nb_probas=nb_probas)
            elif graph_type == "abcd":
                graphs, _, b = abcd_equal_size_range_xi(num_graphs=nb_probas, n=n, K=K)
            for algo in algorithms:
                t = compute_running_time(metric, algo, graphs, b, graph_type, n_runs)
                results[metric].loc[K, algo] = t
        print(f"\nTesting metric = {metric} done!")
    # Save results
    file_path = f"time_evaluations/runtime_vs_K_n{n}.csv"
    for metric, df in results.items():
        df.to_csv(file_path.replace(".csv", f"_{metric}.csv"))
        print(f"\nSaved to {file_path.replace('.csv', f'_{metric}.csv')}")
    if plot:
        for metric, df in results.items():
            plot_runtime_vs_K(df, title=f"Runtime vs Number of Communities ({metric})")
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
