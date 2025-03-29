import numpy as np 
import igraph as ig 
import graph_tool.all as gt

import subprocess
import os 

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
    xi: float=0.2,        # OR set mu instead
    mu: float=None,
    islocal:str="false",
    isCL: str="false",
    nout: int=0,
    degreefile: str="deg.dat",
    communitysizesfile: str="cs.dat",
    communityfile: str="com.dat",
    networkfile: str="edge.dat"
):
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

def load_edge_dat_as_igraph(edge_file: str, zero_index: bool = True):
    edges = []
    with open(edge_file, "r") as f:
        for line in f:
            u, v = map(int, line.strip().split())
            if zero_index:
                u -= 1
                v -= 1
            edges.append((u, v))

    g = ig.Graph(edges=edges)
    print(f"Loaded graph with {g.vcount()} nodes and {g.ecount()} edges")
    return g

def load_communities(graph: ig.Graph, com_file: str, zero_index: bool = True):
    with open(com_file, "r") as f:
        labels = []
        for line in f:
            parts = line.strip().split()
            community = int(parts[1])  
            if zero_index:
                community -= 1
            labels.append(community)
    graph.vs["community"] = labels
    print("Loaded community labels into graph")
    return graph


def abcd_generation(filename: str, K: int=None, d_max: int=50, c_min: int=50, c_max: int=1000, n: int=1000, xi: float=0.2, script_path: str="ABCD/ABCDSampler.jl"):
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
        config_path, n=n, d_max=d_max, c_min=c_min, c_max=c_max, xi=xi,
        degreefile=degree_path, communitysizesfile=cs_path, 
        communityfile=community_path, networkfile=edge_path
    )
    result = subprocess.run(
        ["julia", script_path, config_path],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("Julia CLI graph generation failed:")
        print(result.stderr)
    else:
        print("Graph generation completed!")
        print(result.stdout)
    graph = load_edge_dat_as_igraph(edge_path)
    load_communities(graph, community_path)
    
    graph_gt = gt.Graph(directed=False)
    graph_gt.add_vertex(graph.vcount())
    graph_gt.add_edge_list(graph.get_edgelist())
    
    vprop = graph_gt.new_vertex_property("int")
    for v in graph_gt.vertices():
        vprop[v] = graph.vs["community"][int(v)]
    graph_gt.vp["block"] = vprop

    print("Converted to graph_tool.Graph with community labels.")

    return graph_gt

def abcd_equal_size_range_K(range_K: np.array=None, xi: float=0.2, n: int=1000):
    if range_K is None:
        range_K = [k for k in range(2, min(n // 10 + 1, 50)) if n % k == 0]
    assert len(range_K) >= 1
    
    abcd_graphs = {}
    for i, K in enumerate(range_K):
        print(f"\n=== Generating graph {i} with community size {K} ===")
        graph = abcd_generation(f"graph_{i}_K={K}", K=K, n=n, xi=xi)
        if graph is not None:
            abcd_graphs[f"graph_{i}"] = graph
    from IPython.display import clear_output
    clear_output(wait=True)
    print("Graph generated!")
    return abcd_graphs

def abcd_equal_size_range_xi(range_xi: np.array=None, num_graphs: int=5, n: int=1000, K: int=10):
    assert n%K == 0, f"n={n} must be divisible by K={K} for equal-sized communities."
    if range_xi is None:
        range_xi = np.linspace(0.2, 1, num_graphs, endpoint=True)
    assert len(range_xi) >= 1, "Range of xi must contain at least one value."
    
    abcd_graphs = {}
    for i, xi in enumerate(range_xi):
        print(f"\n=== Generating graph {i} with xi = {xi:.2f} ===")
        graph = abcd_generation(f"graph_{i}_xi={xi:.2f}", K=K, n=n, xi=xi)
        if graph is not None:
            abcd_graphs[f"graph_{i}"] = graph
    
    from IPython.display import clear_output
    clear_output(wait=True)
    print("Graph generated!")
    return abcd_graphs, range_xi

def abcd_equal_size_range_dmax(num_graphs: int=10, n: int=1000, K: int=10, xi: float=0.2):
    assert n%K == 0, f"n={n} must be divisible by K={K} for equal-sized communities."
    range_d = np.linspace(20, n//10, num_graphs, dtype=int)
    assert len(range_d) >= 1, "Range of xi must contain at least one value."
    
    abcd_graphs = {}
    for i, d_max in enumerate(range_d):
        print(f"\n=== Generating graph {i} with d_max = {d_max} ===")
        graph = abcd_generation(f"graph_{i}_dmax={d_max}", K=K, d_max=d_max, n=n, xi=xi)
        if graph is not None:
            abcd_graphs[f"graph_{i}"] = graph
    
    from IPython.display import clear_output
    clear_output(wait=True)
    
    print("Graph generated!")
    return abcd_graphs

def abcd_equal_size_range_c_min(num_graphs: int=10, n: int=1000, K: int=10, xi: float=0.2):
    assert n%K == 0, f"n={n} must be divisible by K={K} for equal-sized communities."
    range_c = np.linspace(20, n//10, num_graphs, dtype=int)
    assert len(range_c) >= 1, "Range of xi must contain at least one value."
    
    abcd_graphs = {}
    for i, c_min in enumerate(range_c):
        print(f"\n=== Generating graph {i} with d_max = {c_min} ===")
        graph = abcd_generation(f"graph_{i}_c_min={c_min}", K=K, c_min=c_min, n=n, xi=xi)
        if graph is not None:
            abcd_graphs[f"graph_{i}"] = graph
    
    from IPython.display import clear_output
    clear_output(wait=True)
    print("Graph generated!")
    return abcd_graphs