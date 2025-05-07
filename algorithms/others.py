import igraph as ig 
import numpy as np 

def walktrap(graph: ig.Graph, K: int, steps: int=4):
    """Compute the walktrap algorithm on a graph.
    Args:
        graph (igraph.Graph): Graph to compute the algorithm on.
    Returns:
        list: List of communities.
    """
    communities = graph.community_walktrap(steps=steps)
    return communities.as_clustering(n =  K)
