import numpy as np 
import graph_tool.all as gt
import igraph as ig 

import subprocess
import os 

## Configuration file generation ##
def generate_script_config(
    path: str,
    seed: int=42,
    n: int=1000,
    t1: int=2.5,
    d_min: int=5,
    d_max: int=50,
    d_max_iter: int=1000,
    t2: int=2,
    c_min: int=50,
    c_max: int=1000,
    c_max_iter: int=1000,
    xi: float=0.2,
    islocal:str="false",
    isCL: str="false",
    nout: int=0,
    degreefile: str="deg.dat",
    communitysizesfile: str="cs.dat",
    communityfile: str="com.dat",
    networkfile: str="edge.dat"
):
    """Creation of configuration file of a given directory, will then be used to generate an ABCD graph. [cf https://github.com/bkamins/ABCDGraphGenerator.jl]

    Args:
        path (str): _description_
        seed (int, optional): Use "" for no seeding. Defaults to 42.
        n (int, optional): Number of vertices in the graph. Defaults to 1000.
        t1 (int, optional): Power law exponent for degree distribution. Defaults to 2.5.
        d_min (int, optional): Minimum degree. Defaults to 5.
        d_max (int, optional): Maximum degree. Defaults to 50.
        d_max_iter (int, optional): Maximum number of iterations for sampling degrees. Defaults to 1000.
        t2 (int, optional): Power law exponent for cluster size distribution. Defaults to 2.
        c_min (int, optional): Minimum cluster size. Defaults to 50.
        c_max (int, optional): Maximum cluster size. Defaults to 1000.
        c_max_iter (int, optional): Maximum number of iterations for sampling cluster sizes. Defaults to 1000.
        xi (float, optional): Fraction of edges to fall in background graph. Defaults to 0.2.
        islocal (str, optional): If "true" mixing parameter is restricted to local cluster, otherwise it is global. Defaults to "false".
        isCL (str, optional): If "false" use configuration model, if "true" use Chung-Lu. Defaults to "false".
        nout (int, optional): If nout is passed and is not zero then we require islocal = "false",
                            isCL = "false", and xi (not mu) must be passed,
                            if nout > 0 then it is recommended that xi > 0. Defaults to 0.
        degreefile (str, optional): Name of file do generate that contains vertex degrees. Defaults to "deg.dat".
        communitysizesfile (str, optional): Name of file do generate that contains community sizes. Defaults to "cs.dat".
        communityfile (str, optional): Name of file do generate that contains assignments of vertices to communities. Defaults to "com.dat".
        networkfile (str, optional): Name of file do generate that contains edges of the generated graph. Defaults to "edge.dat".
    """
    lines = [
        f'seed = "{seed}"',
        f'n = "{n}"',
        f't1 = "{t1}"',
        f'd_min = "{d_min}"',
        f'd_max = "{d_max}"',
        f'd_max_iter = "{d_max_iter}"',
        f't2 = "{t2}"',
        f'c_min = "{c_min}"',
        f'c_max = "{c_max}"',
        f'c_max_iter = "{c_max_iter}"',
    ]

    if xi is not None:
        lines.append(f'xi = "{xi}"')
    elif mu is not None:
        lines.append(f'mu = "{mu}"')

    lines += [
        f'islocal = "{islocal}"',
        f'isCL = "{isCL}"',
        f'degreefile = "{degreefile}"',
        f'communitysizesfile = "{communitysizesfile}"',
        f'communityfile = "{communityfile}"',
        f'networkfile = "{networkfile}"',
        f'nout = "{nout}"'
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))

## Graph generation ##
def load_edge_dat(edge_file: str, zero_index: bool = True) -> ig.Graph:
    edges = np.loadtxt(edge_file, dtype=int)
    if zero_index:
        edges -= 1
    num_vertices = edges.max() + 1
    g = ig.Graph(edges.tolist(), directed=False)
    g.add_vertices(num_vertices - g.vcount())
    return g

def load_communities(graph: ig.Graph, com_file: str, zero_index: bool = True) -> list:
    """
    Load community assignments from file and attach as a vertex attribute to an igraph.Graph.
    
    Parameters:
    - graph (igraph.Graph): The graph to annotate.
    - com_file (str): Path to the community membership file.
    - zero_index (bool): Whether to adjust indexing to zero-based.
    
    Returns:
    - membership (list): A list of community assignments indexed by vertex ID.
    """
    data = np.loadtxt(com_file, dtype=int)
    vertex_ids = data[:, 0]
    communities = data[:, 1]
    if zero_index:
        vertex_ids -= 1

    membership = [None] * graph.vcount()
    for vid, comm in zip(vertex_ids, communities):
        membership[vid] = comm

    graph.vs["community"] = membership
    return membership

def abcd_generation(filename: str, K: int=None, d_min: int=5, d_max: int=50, c_min: int=50, c_max: int=1000, n: int=3000, xi: float=0.2, script_path: str="ABCD/ABCDSampler.jl"):
    """Generate an ABCD graph using the Julia ABCDGraphGenerator.jl package.
    The graph is generated based on the specified parameters and saved to the specified directory.

    Args:
        filename (str): Name of the output directory and configuration file.
        K (int, optional): Number of communities. If None, the number of communities is determined by c_min and c_max. Defaults to None.
        d_min (int, optional): Minimum degree. Defaults to 5.
        d_max (int, optional): Maximum degree. Defaults to 50.
        c_min (int, optional): Minimum community size. Defaults to 50.
        c_max (int, optional): Maximum community size. Defaults to 1000.
        n (int, optional): Number of vertices in the graph. Defaults to 1000.
        xi (float, optional): Fraction of edges to fall in background graph. Defaults to 0.2.
        script_path (str, optional): Path to the Julia script for graph generation. Defaults to "ABCD/ABCDSampler.jl".

    Returns:
        graph (gt.Graph): The generated graph as a graph-tool object.
    """
    if K is not None:
        assert n%K == 0, f"n={n} must be divisible by K={K} for equal-sized communities."
        c_min = int(n/K)
        c_max = int(n/K)
        output_dir = "ABCD/conf/same_size/"
    else:
        if c_min == c_max: # communities of same size
            assert n % c_min == 0, f"n={n} must be divisible by c_min={c_min} for equal-sized communities."
            output_dir = "ABCD/conf/same_size/"
            if K is not None:
                assert n%K == 0, f"n={n} must be divisible by K={K} for equal-sized communities."
                c_min = int(n / K)
                c_max = int(n / K)
        else: # communities of different size
            output_dir = "ABCD/conf/varying_size/"
    
    output_dir_ = output_dir + filename
    os.makedirs(output_dir_, exist_ok=True)

    config_path = os.path.join(output_dir_, f"{filename}.toml")
    edge_path = os.path.join(output_dir_, "edge.dat")
    degree_path = os.path.join(output_dir_, "degree.dat")
    cs_path = os.path.join(output_dir_, "cs.dat")
    community_path = os.path.join(output_dir_, "com.dat")
    
    generate_script_config(
        config_path, n=n, d_min=d_min, d_max=d_max, c_min=c_min, c_max=c_max, xi=xi,
        degreefile=degree_path, communitysizesfile=cs_path, 
        communityfile=community_path, networkfile=edge_path
    )
    result = subprocess.run(
        ["julia", script_path, config_path],
        capture_output=True,
        text=True
    )
    
    graph = load_edge_dat(edge_path)
    block_membership=load_communities(graph, community_path)

    return graph, block_membership

def abcd_equal_size_range_K(range_K: np.array=None, xi: float=0.4, n: int=3000, n_graphs: int=5):
    """Generate a range of ABCD graphs with equal-sized communities.
    The number of communities is determined by the range_K parameter.
    The graphs are generated based on the specified parameters and saved to the specified directory.
    Args:
        range_K (np.array, optional): Range of community sizes. If None, the range is determined based on n. Defaults to None.
        xi (float, optional): Fraction of edges to fall in background graph. Defaults to 0.4.
        n (int, optional): Number of vertices in the graph. Defaults to 3000.
    Returns:
    """
    if range_K is None:
        range_K = [k for k in range(2, min(n // 10 + 1, 50)) if n % k == 0]
    assert len(range_K) >= 1
    
    abcd_graphs = {}
    memberships = {}
    for i, K in enumerate(range_K):
        for rep in range(n_graphs):
            name = f"ABCD_K{K}_rep{rep}"
            graph, b = abcd_generation(f"graph_{i}_K={K}", K=K, n=n, xi=xi)
            if graph is not None:
                abcd_graphs[name] = graph
                memberships[name] = b
    print("Graph generated!")
    return abcd_graphs, memberships

def abcd_equal_size_range_xi(range_xi: np.array=None, num_graphs: int=5, xi_max: float = 1, n: int=1000, K: int=10, c_min: int=50, c_max:int=1000, d_min:int=5, d_max: int=50):
    """Generate a range of ABCD graphs with equal-sized communities.
    The number of communities is determined by the range_xi parameter.

    Args:
        range_xi (np.array, optional): Range of xi values. If None, the range is determined based on num_graphs. Defaults to None.
        num_graphs (int, optional): Number of graphs to generate. Defaults to 5.
        xi_max (float, optional): Maximum value of xi. Defaults to 1.
        n (int, optional): Number of vertices in the graph. Defaults to 1000.
        K (int, optional): Number of communities. Defaults to 10.
        c_min (int, optional): Minimum community size. Defaults to 50.
        c_max (int, optional): Maximum community size. Defaults to 1000.
        d_min (int, optional): Minimum degree. Defaults to 5.
        d_max (int, optional): Maximum degree. Defaults to 50.

    Returns:
        abcd_graphs (dict): Dictionary of generated graphs.
        range_xi (np.array): Range of xi values used for graph generation
    """
    assert n%K == 0, f"n={n} must be divisible by K={K} for equal-sized communities."
    assert xi_max > 0 and xi_max <=1, "xi has to be between 0 and 1"
    if range_xi is None:
        range_xi = np.linspace(0.1, xi_max, num_graphs, endpoint=True)
    assert len(range_xi) >= 1, "Range of xi must contain at least one value."
    
    abcd_graphs = {}
    memberships = {}
    for i, xi in enumerate(range_xi):
        graph, b = abcd_generation(f"graph_{i}_xi={xi:.2f}", K=K, d_min=d_min, d_max=d_max, c_min=c_min, c_max=c_max, n=n, xi=xi)
        if graph is not None:
            abcd_graphs[f"graph_{i}"] = graph
            memberships[f"graph_{i}"] = b
    
    from IPython.display import clear_output
    clear_output(wait=True)
    return abcd_graphs, range_xi, memberships
