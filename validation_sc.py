import numpy as np 
import pandas as pd 
import igraph as ig
import networkx as nx
import graph_tool.all as gt
from validation import *

from SBM.sbm_generation import sbm_generation
from ABCD.abcd_generation import *

algorithms = ["bayesian", "spectral", "leiden", "louvain", "walktrap"]
algo_fixed_K = ["spectral", "walktrap"]
algo_not_fixed_K = [algo for algo in algorithms if algo not in algo_fixed_K]
possible_metrics = [
    "adjusted_mutual_info_score", 
    "adjusted_rand_score", 
    "v_measure_score",
    "homogeneity_score",
    "fowlkes_mallows_score"
]
graph_types = ["sbm", "abcd"]