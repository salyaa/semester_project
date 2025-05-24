import numpy as np 
import pandas as pd 
import igraph as ig
import networkx as nx
import graph_tool.all as gt
from validation import *

from SBM.sbm_generation import sbm_generation
from ABCD.abcd_generation import *
from helpers import *
from validation import *

from constants import algorithms_sc, possible_metrics, graph_types

possible_algos = algorithms_sc

def compute_score_sc(metric, algorithm, graphs, block_membership, graph_type="sbm"):
    """
    Compute the score for the given algorithm and metric.
    """
    assert algorithm in possible_algos, "The algorithm must be one that we implemented."
    assert graph_type in graph_types, "The type of graph must be one that we implemented."
    assert metric in possible_metrics, "The metric must be one that we implemented."
    
    memberships, true_clusters = get_predicted_memberships(algorithm, graphs, block_membership, possible_algos, graph_type)
    
    metric_functions = {
        "adjusted_mutual_info_score": adjusted_mutual_info_score,
        "adjusted_rand_score": adjusted_rand_score,
        "v_measure_score": v_measure_score, #identical to normalized_mutual_info_score with the 'arithmetic' option for averaging.
        "homogeneity_score": homogeneity_score,
        "fowlkes_mallows_score": fowlkes_mallows_score
    }
    compute_metric = metric_functions[metric] 
    
    scores = dict()
    for key, _ in graphs.items():
        scores[key] = compute_metric(true_clusters[key], memberships[key])
    number_clusters_found = {key: len(set(memberships[key])) for key in memberships.keys()}
    
    avg_score = np.mean(list(scores.values()))
    std_score = np.std(list(scores.values()))
    
    return scores, avg_score, std_score, number_clusters_found