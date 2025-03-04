import graph_tool.all as gt 
import numpy as np

def sbm_generation(n: int = 100, K: int = 2, p: float = 0.5, nb_probas: int = 8, modify: str = "out"):
    """
    Generate multiple SBM graphs while modifying either p_in or p_out.

    Parameters:
    - n (int): Number of nodes.
    - K (int): Number of communities.
    - p (float): Base probability for edges (p_in or p_out).
    - nb_probas (int): Number of different probabilities to test.
    - modify (str): Define which probability to change ("in" for p_in, "out" for p_out).

    Returns:
    - graphs (dict): A dictionary containing SBM graphs with different p values.
    """
    
    assert 0 <= p <= 1, "p must be between 0 and 1"
    assert modify in ["in", "out"], "modify must be 'in' or 'out'"

    group_sizes = [n // K] * K
    remainder = n % K  
    for i in range(remainder):
        group_sizes[i] += 1
    
    range_p = np.linspace(0, 1, num=nb_probas, endpoint=False)
    
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

    return graphs
