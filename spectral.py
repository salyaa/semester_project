import networkx as nx
from sklearn.cluster import SpectralClustering
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

def plot_clust(G: nx.Graph, clusters):
    """Plot the graph with nodes colored by their cluster assignments.

    Args:
        G (nx.Graph): The input graph.
        clusters (list): List of cluster labels for each node.
    """
    # Choose a layout for the graph (spring layout for better spacing)
    pos = nx.spring_layout(G, seed=42)

    # Get unique clusters and assign colors
    unique_clusters = np.unique(clusters)
    num_clusters = len(unique_clusters)
    colors = plt.cm.get_cmap("viridis", num_clusters)  # Color map

    plt.figure(figsize=(8, 8))
    for i, cluster in enumerate(unique_clusters):
        nodes = [node for node, label in enumerate(clusters) if label == cluster]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=[colors(i)], node_size=100)
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=0.5)
    plt.axis("off")
    plt.show()


def spectral(G: nx.Graph, K: int, plot: bool=True):
    """Perform Spectral Clustering on a graph.

    Args:
        G (nx.Graph): The input graph.
        K (int): The number of clusters.

    Returns:
        list: Cluster labels for each node.
    """
    # Convert graph to adjacency matrix
    A = nx.adjacency_matrix(G).astype(np.float32)

    # Convert to sparse format with the correct dtype
    A = csr_matrix(A, dtype=np.float32)

    # Perform spectral clustering
    spectral_clustering = SpectralClustering(n_clusters=K, n_init=100, affinity="precomputed", random_state=42)
    spectral_clustering.fit(A)

    # Get cluster assignments
    clusters = spectral_clustering.labels_
    
    if plot:
        plot_clust(G, clusters)

    return clusters

## Implement one using the non-backtracking matrix? normalized Laplacian?