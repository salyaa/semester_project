**Benchmarking Community Detection Algorithm**

1.  Background and Motivation:

    Community detection is a critical problem in network science and graph theory, aiming to uncover
    the underlying structure of complex networks by identifying groups of nodes that are more densely
    connected internally than with the rest of the network. Applications span diverse fields such as
    social network analysis, biology, recommendation systems, and more.
    Despite the large number of community detection algorithms available, there is no one-size-fits-all
    solution. The performance of these algorithms can vary widely depending on factors such as
    network size, density, and structure. Systematic benchmarking provides insights into their strengths
    and limitations, guiding the choice of algorithms for specific applications.
    This project aims to provide a comprehensive benchmarking study of popular community detection
    algorithms using both synthetic and real-world networks, focusing on performance metrics such as
    accuracy, scalability, robustness, and interpretability.

2.  Objectives:

    The main objectives of the project are as follows:

    1. Survey and Categorize Algorithms: Review state-of-the-art community detection
       algorithms and categorize them based on methodologies (e.g., modularity optimization, spectral clustering, probabilistic methods, etc.). Starting points: [1,2,3]
    2. Develop a Benchmarking Framework: Design and implement a robust framework for
       testing community detection algorithms across various datasets and performance metrics.
    3. Evaluate Performance:
       - accuracy
       - scalability
       - robustness
       - practicality
    4. Synthesize Results: Provide a comparative analysis, identifying trade-offs and best-fit scenarios for different algorithms.
    5. Tool Development (Optional): Package the benchmarking framework as an open-source tool to facilitate future research.

3.  Methodology:
    3.1. Algorithm Selection:

         Include a diverse set of algorithms from different categories, such as:
             - Modularity-based (Louvain, Leiden, Bayan)
             - Spectral methods (several variants of Spectral Clustering)
             - Statistical Models --> SBM (algorithms: Bayesian inference, Variational EM, pseudo-likelihood maximization)
             - Label Propagation (LPA)
             - Deep learning based (GNN, e.g., DGI, GraphSAGE for clustering)
             - Others (Infomap, Walktrap, Semi-definite programming)

    3.2. Dataset Preparation:

         - Synthetic datasets: generate networks with ground-truth communities using models like SBM or ABCD;
         - Real-World datasets: select datasets with known community structures from repositories like Network Repository or SNAP.

    3.3. Benchmarking metrics:

        - Accuracy: NMI, ARI, F1-score;
        - Efficiency: Runtime and memory usage;
        - Robustness: Performance under noise (edge addition or removal);
        - Applicability: Qualitative analysis of real-world interpretability;
        - Theoretic guarantees.

**References:**

[1] Fortunato, Santo, and Darko Hric. "Community detection in networks: A user guide." Physics
reports 659 (2016): 1-44.

[2] Avrachenkov, Konstantin, and Maximilien Dreveton. Statistical Analysis of Networks. Now
Publishers, 2022.

[3] T. P. Peixoto, Descriptive Vs. Inferential Community Detection in Networks: Pitfalls, Myths and
Half-Truths, Elements in the Structure and Dynamics of Complex Networks (2023).

[4] https://graph-tool.skewed.de/

[5] https://github.com/GiulioRossetti/cdlib
