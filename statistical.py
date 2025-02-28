import graph_tool as gt
import numpy as np 


## Bayesian inference
# Use graph-tool 
# (https://graph-tool.skewed.de/static/doc/demos/inference/inference.html)
# OR Use PyMC / networkx
def bayesianInf(G: gt.collection, plot: bool=True):
    #fit the degree-corrected model
    state = gt.minimize_blockmodel_dl(G) #returns a BlockState object that includes the inference results
    
    if plot:
        state.draw(pos=g.vp.pos)
    
    clusters = state.get_blocks()
    
    return clusters

# def mcmc(G: gt.collection, plot: bool=True):
#     state = gt.minimize_nested_blockmodel_dl(G)

#     # S1 = state.entropy()
#     # for i in range(1000): # this should be sufficiently large
#     #     state.multiflip_mcmc_sweep(beta=np.inf, niter=10)
#     # S2 = state.entropy()
#     # print("Improvement:", S2 - S1)
    
#     gt.mcmc_anneal(state, beta_range=(1, 10), niter=1000, mcmc_equilibrate_args=dict(force_niter=10))
    

## Variation EM
# def variationalEM(G: ig.Graph):
#     return 0