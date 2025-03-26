import graph_tool.all as gt 
import numpy as np

def sbm_generation(n: int = 500, K: int = 3, nb_probas: int = 5, range_p: np.array=None, modify: str = "out"):
    """
    Generate multiple SBM graphs while modifying either p_in or p_out.

    Parameters:
    - n (int): Number of nodes.
    - K (int): Number of communities.
    - p (float): Base probability for edges (p_in or p_out).
    - nb_probas (int): Number of different probabilities to test, i.e., number of generated graphs.
    - modify (str): Define which probability to change ("in" for p_in, "out" for p_out).

    Returns:
    - graphs (dict): A dictionary containing SBM graphs with different p values.
    """
    
    assert modify in ["in", "out"], "modify must be 'in' or 'out'"
    assert K>2, "need at least 3 clusters"
    assert n > K
    
    p = np.log(n)/n

    ## Distribute evenly nodes across communities
    group_sizes = [n // K] * K
    remainder = n % K  
    for i in range(remainder):
        group_sizes[i] += 1
    
    if range_p is None:
        if nb_probas == 1:
            range_p = [np.log(n)/n]
        elif nb_probas == 2:
            range_p = [np.log(n)/(3*n), np.log(n)/(2*n)]
        else:
            range_p = np.linspace(0, p, num=nb_probas, endpoint=True)
    
    graphs = {}
    for i, p_mod in enumerate(range_p):
        if modify == "out":
            communities_matrix = np.full((K, K), p_mod) 
            np.fill_diagonal(communities_matrix, p) 
        elif modify == "in":
            communities_matrix = np.full((K, K), p)  
            np.fill_diagonal(communities_matrix, p_mod) 

        block_membership = np.concatenate([[k] * size for k, size in enumerate(group_sizes)])

        G, b = gt.random_graph(n, 
                            lambda : (np.random.poisson(5), np.random.poisson(5)),
                            model="blockmodel",
                            edge_probs=communities_matrix,
                            block_membership=block_membership)

        G.vp["block"] = b
        graphs[f"SBM_{modify}_{i}"] = G  
    return graphs, range_p

## Could create a function to plots the various graph to see the impact of modifying p_in or p_out.
