import graph_tool.all as gt
import numpy as np 
import matplotlib.pyplot as plt


## Bayesian inference
# Use graph-tool 
# (https://graph-tool.skewed.de/static/doc/demos/inference/inference.html)
# OR Use PyMC / networkx
def bayesianInf(G: gt.Graph, seed: int = 42, plot: bool = False):
    """
    Perform Bayesian inference on a graph using a degree-corrected stochastic block model (DC-SBM).
    
    Parameters:
    - G (gt.Graph): The input graph.
    - seed (int): To control the randomness of mcmc_sweep()
    - plot (bool): Whether to visualize the inferred clusters. Default is True.
    
    Returns:
    - state (gt.BlockState): The fitted block model containing clustering information.
    - clusters (gt.PropertyMap): A mapping of nodes to inferred community labels.
    """
    if seed is not None:
        np.random.seed(seed)  # Fixes randomness for inference

    state = gt.minimize_blockmodel_dl(G, state_args=dict(deg_corr=True))
    state.mcmc_sweep(niter=000)
    clusters = state.get_blocks()
    
    if plot and "pos" in G.vp:
        cluster_colors = G.new_vertex_property("vector<double>")
        
        unique_clusters = list(set(clusters.a))
        num_clusters = len(unique_clusters)
        colormap = plt.cm.get_cmap("spring", num_clusters) 
        
        for v in G.vertices():
            cluster_index = unique_clusters.index(clusters[v]) 
            cluster_colors[v] = colormap(cluster_index)[:3] 
        
        node_size = G.new_vertex_property("double", val=10) 

        state.draw(pos=G.vp.pos, 
                vertex_fill_color=cluster_colors, 
                vertex_size=node_size,
                edge_pen_width=0.7,
                output_size=(600, 600))

    num_clusters = len(set(clusters))
    
    return clusters
