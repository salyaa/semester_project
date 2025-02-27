import igraph as ig 
import matplotlib.pyplot as plt
import louvain as lv

# Create the pipeline for the modularity based community detection

# If we want to plot the graph with the different clusters, use this:
def plot_clusters(G: ig.Graph, clusters):
    layout = G.layout('kk')
    K = len(clusters)
    palette = ig.drawing.colors.ClusterColoringPalette(K)
    colors = [palette[i] for i in clusters.membership]
    
    _, ax = plt.subplots(figsize=(8, 8))
    ig.plot(
        clusters, 
        layout=layout, 
        vertex_color=colors, 
        vertex_size=10, 
        edge_width=0.5,
        bbox=(500, 500),
        margin=20,
        target=ax 
    )
    plt.show()


def louvain(G: ig.Graph, plot : bool = True):
    # https://github.com/vtraag/louvain-igraph?tab=readme-ov-file
    """Find the communities of an igraph graph G using the Louvain algorithm

    Args:
        G (igraph.Graph)
        plot (boolean): if true, plot the graph with different colors for the clusters
    
    Return:
        list: A list of VertexClustering objects, representing the communities
    """
    clusters = G.community_multilevel()
    #clusters = lv.find_partition(G, lv.ModularityVertexPartition) 
    # Obtain almost the same results for Karate Club graph and ER graph 
    if plot:
        plot_clusters(G, clusters)
    return clusters


def leiden(G: ig.Graph, plot: bool=True):
    """Find the communities of an igraph graph G using the Leiden algorithm

    Args:
        G (igraph.Graph)
    
    Return:
        VertexClustering object from igraph
    """
    #clusters = ig.Graph.community_leiden(G)
    clusters = G.community_leiden()
    
    if plot:
        plot_clusters(G, clusters)
    
    return clusters


#def bayan(G):
