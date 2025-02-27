import igraph as ig 


## Bayesian inference
def bayesianInf(G):
    return 0

## Variation EM
def variationalEM(G: ig.Graph):
    sbm = G.community_infomap() # fit an SBM using Variational EM
    clusters = sbm.membership
    
    return clusters
