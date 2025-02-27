import igraph as ig 


## Bayesian inference
def bayesianInf(G):
    return 0

## Variation EM
def variationalEM(G: ig.Graph):
    sbm = G.community_infomap() #fit SBM using VEM
    clusters = sbm.membership
    
    return clusters
