import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.rcParams.update({
    "axes.titlesize":   16,  # subplot titles
    "axes.labelsize":   16,  # x/y labels
    "xtick.labelsize":  12,  # x tick labels
    "ytick.labelsize":  12,  # y tick labels
    "legend.fontsize":  12,  # legend text
    "legend.title_fontsize": 13,
    "figure.titlesize": 17   # suptitle
})

from constants import *

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
    results_sem,
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

    metrics     = results_mean.index.get_level_values("Metric").unique()
    param_vals  = results_mean.index.get_level_values(param_name).unique()

    fig, axs = plt.subplots(
      2, len(metrics),
      figsize=(5*len(metrics), 10),
      sharex=True, sharey=True
    )
    if axs.ndim == 1:
        axs = np.array([[axs[0]], [axs[1]]])

    for col, metric in enumerate(metrics):
        means = results_mean.xs(metric, level="Metric").astype(float)
        sems  = results_sem.xs(metric, level="Metric").astype(float)

        for row, algos in enumerate([algo_not_fixed_K, algo_fixed_K]):
            ax = axs[row, col]
            for algo in algos:
                m = means[algo]
                s = sems[algo]
                ax.plot(param_vals, m, '-x', label=algo)
                ax.fill_between(param_vals, m-s, m+s, alpha=0.2)

            ax.set_title(
                f"{metric.replace('_',' ').title()} "
                f"({'Known K' if row==1 else 'Inferred K'})",
                fontsize=16, fontweight="bold"
            )
            if row == 1:
                ax.set_xlabel(param_label or param_name, fontsize=14, labelpad=6)
            if col == 0:
                ax.set_ylabel("Score", fontsize=14, labelpad=6)

            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.grid(True)
            ax.legend(fontsize=12, title_fontsize=13)

    fig.suptitle(title, fontsize=18, fontweight="bold")
    plt.tight_layout(rect=[0,0.03,1,0.95])
    plt.show()

def plot_results_generic_sc(
    results_mean: pd.DataFrame,
    results_sem:  pd.DataFrame,
    param_name:   str,
    param_label:  str = None,
    title:        str = None
):
    """
    Plots mean±SEM for a custom list of algorithms (e.g. your SC variants)
    over a single row of subplots (one column per metric).
    """
    from IPython.display import clear_output
    clear_output(wait=True)

    metrics    = results_mean.index.get_level_values("Metric").unique()
    param_vals = results_mean.index.get_level_values(param_name).unique()

    # One row, M columns (one per metric)
    fig, axs = plt.subplots(
        1, len(metrics),
        figsize=(5*len(metrics), 5),
        sharex=True, sharey=True
    )
    if len(metrics) == 1:
        axs = [axs]

    for ax, metric in zip(axs, metrics):
        means = results_mean.xs(metric, level="Metric").astype(float)
        sems  = results_sem.xs(metric,  level="Metric").astype(float)

        for algo in algorithms_sc:
            if algo not in means.columns:
                raise ValueError(f"Algorithm '{algo}' not found in results for metric '{metric}'.")
            m = means[algo]
            s = sems[algo]
            ax.plot(param_vals, m, '-x', label=algo)
            ax.fill_between(param_vals, m-s, m+s, alpha=0.2)

        ax.set_title(metric.replace('_',' ').title(), fontsize=16, fontweight="bold")
        ax.set_xlabel(param_label or param_name, fontsize=14)
        ax.set_ylabel("Score", fontsize=14)
        ax.grid(True)
        ax.legend(fontsize=12, title="Algorithm", title_fontsize=13)
        ax.tick_params(labelsize=12)

    fig.suptitle(title, fontsize=18, fontweight="bold")
    plt.tight_layout(rect=[0,0.03,1,0.95])
    plt.show()



def plot_cluster_size_boxplots(
    cluster_sizes,      # dict[param → dict[algo → list of inferred sizes]]
    true_cluster_sizes,# dict[param → list of true sizes]
    param_range, 
    param_name: str,
    title: str = "Distribution of cluster sizes"
):
    fig, axes = plt.subplots(
        len(algorithms), 1,
        figsize=(6, 3*len(algorithms)),
        sharex=True
    )

    for ax, algo in zip(axes, algorithms):
        # 1) inferred‐sizes boxplot
        data = [ cluster_sizes[p][algo] for p in param_range ]
        width = (param_range[1]-param_range[0])*0.8 if len(param_range)>1 else 0.6
        bp = ax.boxplot(
            data,
            positions=param_range,
            widths=width,
            patch_artist=True,
            boxprops=dict(facecolor="C0", alpha=0.6),  # blue boxes
            medianprops=dict(color="navy")
        )

        # 2) overlay true sizes as red dots
        for p in param_range:
            ts = true_cluster_sizes[p]
            x  = np.full(len(ts), p)
            ax.scatter(
                x, ts,
                color="red",
                alpha=0.6,
                s=20,
                label="True size" if algo==algorithms[0] else ""
            )

        ax.set_ylabel(f"{algo}\ncluster size", fontsize=12)
        ax.grid(axis='y')

    axes[-1].set_xlabel(param_name, fontsize=14)
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # only one legend
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, loc="upper right")

    plt.tight_layout(rect=[0,0.03,1,0.95])
    plt.show()

def plot_single_score(
    results_mean: pd.DataFrame,
    results_std: pd.DataFrame,
    param_name: str,
    metric: str="v_measure_score",
    param_label: str = None,
    title: str = None
):
    """
    Plot a single metric across varying parameter values, showing fixed-K and inferred-K separately.

    Args:
        metric (str): The metric to plot (e.g., "adjusted_rand_score").
        results_mean (pd.DataFrame): Mean scores indexed by (param, "Metric").
        results_std (pd.DataFrame): Standard deviations indexed similarly.
        param_name (str): The parameter being varied (e.g., "K", "xi").
        param_label (str): Label for x-axis (optional).
        title (str): Title of the plot.
    """
    fixed_K_algos = ["spectral", "walktrap"]
    all_algos = results_mean.columns
    non_fixed_K_algos = [a for a in all_algos if a not in fixed_K_algos]

    param_values = results_mean.index.get_level_values(param_name).unique()

    mean_df = results_mean.xs(metric, level="Metric")
    std_df = results_std.xs(metric, level="Metric")

    fig, axs = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

    for row, algo_group in enumerate([non_fixed_K_algos, fixed_K_algos]):
        ax = axs[row]
        for algo in algo_group:
            mean = mean_df[algo].astype(float)
            std = std_df[algo].astype(float)
            std = np.nan_to_num(std, nan=0.0)

            ax.plot(param_values, mean, label=algo, marker="o")
            ax.fill_between(param_values, mean - std, mean + std, alpha=0.2)

        ax.set_ylabel("Score")
        ax.grid(True)
        ax.set_title(f"{metric.replace('_', ' ').title()} ({'Inferred K' if row == 0 else 'Known K'})")
        ax.legend()

    axs[1].set_xlabel(param_label if param_label else param_name)
    fig.suptitle(title or f"Evaluation of {metric.replace('_', ' ').title()}", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
