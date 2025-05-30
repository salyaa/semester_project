from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, v_measure_score, homogeneity_score, fowlkes_mallows_score

algorithms = ["bayesian", "spectral", "leiden", "louvain", "walktrap"]
algorithms_sc = ["spectral", "sc_bm", "sc_dcbm", "sc_pabm", "orth_sc"]
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

metric_functions = {
        "adjusted_mutual_info_score": adjusted_mutual_info_score,
        "adjusted_rand_score": adjusted_rand_score,
        "v_measure_score": v_measure_score,
        "homogeneity_score": homogeneity_score,
        "fowlkes_mallows_score": fowlkes_mallows_score
    }
