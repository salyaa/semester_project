import graph_tool.all as gt 
import numpy as np

# def sbm_generation_in(n: int = 100, K: int = 2, p: float = 0.5, nb_probas: int = 5):
#     assert p <= 1 and p >= 0 # check that p is a probability
    
#     group_sizes = [n // K] * K
#     remainder = n % K
    
#     for i in range(remainder):
#         group_sizes[i] += 1
    
#     p_outs = np.linspace(0, 1, num = nb_probas, endpoint=False)
#     graphs = dict()
    
#     for i, p_out in enumerate(p_outs):
#         communities_matrix = np.full((K,K), p_out)
#         np.fill_diagonal(communities_matrix, p)

#         G, b = gt.random_graph(n, 
#                             lambda i, j, _: communities_matrix[b[i], b[j]], 
#                             directed=False, 
#                             vertex_distribution=group_sizes)
#         G.vp["block"] = b
#         graphs[f"SBM_in_{i}"] = G
#     return graphs

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

    # Step 1: Compute community sizes
    group_sizes = [n // K] * K
    remainder = n % K  
    for i in range(remainder):
        group_sizes[i] += 1
    
    # Step 2: Define the probability range
    range_p = np.linspace(0, 1, num=nb_probas, endpoint=False)
    
    graphs = {}

    for i, p_mod in enumerate(range_p):
        # Create a KxK probability matrix
        if modify == "out":
            communities_matrix = np.full((K, K), p_mod)  # Modify p_out
            np.fill_diagonal(communities_matrix, p)  # Keep p_in fixed
        elif modify == "in":
            communities_matrix = np.full((K, K), p)  # Keep p_out fixed
            np.fill_diagonal(communities_matrix, p_mod)  # Modify p_in

        # Create block membership property
        block_membership = np.concatenate([[k] * size for k, size in enumerate(group_sizes)])

        # Step 3: Generate SBM graph using model="blockmodel"
        G, b = gt.random_graph(n, 
                               lambda : (np.random.poisson(5), np.random.poisson(5)),  # FIX: Correct (in-degree, out-degree)
                               model="blockmodel",
                               edge_probs=communities_matrix,
                               block_membership=block_membership)  # FIX: Explicitly assign block memberships
        
        # Store community assignments
        G.vp["block"] = b
        
        # Store graph in dictionary
        graphs[f"SBM_{modify}_{i}"] = G  
    
    return graphs
