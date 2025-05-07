#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 11:34:30 2025

@author: dreveton
"""

import numpy as np
import scipy as sp
from sklearn.cluster import KMeans, SpectralClustering

import selfrepresentation as selfrepresentation



# =============================================================================
# SPECTRAL CLUSTERING: VARIOUS FORMS
# =============================================================================


def spectralClustering_bm( A , n_clusters ):
    """ Perform spectral clustering for a SBM
    Parameters
    ----------
    A : {array-like, sparse matrix} of shape (n, n)
        Adjacency matrix of a graph.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    z : vector of shape (n) with integer entries between 1 and n_clusters
        entries z_i denotes the cluster of vertex i.
    """
    n = A.shape[0]
    vals, vecs = sp.sparse.linalg.eigsh( A.astype(float), k = n_clusters, which = 'LM' )
    hatP = vecs @ np.diag( vals ) #@ vecs.T #Note: k-means on vecs @ np.diag( vals ) and on vecs @ np.diag( vals ) @ vecs.T is equivalent, but faster using vecs @ np.diag( vals )  (n-by-n_clusters matrix instead of n-by-n)
    z = KMeans( n_clusters = n_clusters, n_init = 'auto' ).fit_predict( hatP ) + np.ones( n ) 
    
    return z.astype(int) 


def spectralClustering_dcbm( A , n_clusters ):
    """ Perform spectral clustering for a DCBM
    Algorithm from 
    Chao Gao, Zongming Ma, Anderson Y. Zhang, Harrison H. Zhou 
    "Community detection in degree-corrected block models" 
    The Annals of Statistics, Ann. Statist. 46(5), 2153-2185 (2018)

    
    Parameters
    ----------
    A : {array-like, sparse matrix} of shape (n, n)
        Adjacency matrix of a graph.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    z : vector of shape (n) with integer entries between 1 and n_clusters
        entries z_i denotes the cluster of vertex i.
    """
    n = A.shape[0]
    vals, vecs = sp.sparse.linalg.eigsh( A.astype(float), k = n_clusters, which = 'LM' )

    hatP = vecs @ np.diag( vals ) @ vecs.T
    hatP_rowNormalized = hatP
    for i in range( n ):
        if np.linalg.norm( hatP[i,:], ord = 1) != 0:
            hatP_rowNormalized[i,:] = hatP[i,:] / np.linalg.norm( hatP[i,:], ord = 1)
    
    z = KMeans(n_clusters = n_clusters, n_init = 'auto' ).fit_predict( hatP_rowNormalized ) + np.ones( n )
    
    return z.astype(int) 


def spectralClustering_pabm( A, n_clusters, version = 'subspace' ):
    """ Perform spectral clustering for a PABM
    Parameters
    ----------
    A : {array-like, sparse matrix} of shape (n, n)
        Adjacency matrix of a graph.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    z : vector of shape (n) with integer entries between 1 and n_clusters
        entries z_i denotes the cluster of vertex i.
    """
    
    n = A.shape[0]
    
    rank = n_clusters * n_clusters
    
    vals, vecs = sp.sparse.linalg.eigsh( A.astype(float), k = rank, which = 'LM' )
    
    if version == 'kmeans':
        z = KMeans(n_clusters = n_clusters, n_init = 'auto' ).fit_predict( vecs @ np.diag( vals ) ) + np.ones( n )
    
    elif version == 'subspace':
        model = selfrepresentation.ElasticNetSubspaceClustering( n_clusters = n_clusters ,algorithm = 'lasso_lars',gamma=50 ).fit( vecs @ np.diag( vals ) )
        z = model.labels_ + np.ones( n )

    elif version == 'subspace-omp':
        model = selfrepresentation.SparseSubspaceClusteringOMP( n_clusters = n_clusters, thr=1e-5 ).fit( vecs @ np.diag( vals ) )
        z = model.labels_ + np.ones( n )
    
    return z.astype(int)



def orthogonalSpectralClustering( A, n_clusters ):
    """Perform Orthogonal Spectral Clustering. 
    See Algorithm 1 of : 
    John Koo, Minh Tang, and Michael W. Trosset. "Popularity adjusted block models are generalized random dot product graphs." Journal of Computational and Graphical Statistics 32.1 (2023): 131-144.
    
    Parameters
    ----------
    A : {array-like, sparse matrix} of shape (n, n)
        Adjacency matrix of a graph.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    z : vector of shape (n) with integer entries between 1 and n_clusters
        entries z_i denotes the cluster of vertex i.
    """
    n = A.shape[0]

    vals, vecs = sp.sparse.linalg.eigsh( A.astype(float), k = n_clusters * n_clusters, which = 'BE' )
    
    B = np.sqrt( n ) * vecs @ vecs.T
    
    clustering_ = SpectralClustering( n_clusters = n_clusters, affinity='precomputed').fit( np.abs(B) )
    z = clustering_.labels_ + np.ones( n )

    return z.astype(int) 


import math as math

def fast_spectral_cluster( A, n_clusters: int):
    """
    This is a faster spectral clustering, from 
    Peter Macgregor. "Fast and simple spectral clustering in theory and practice." Advances in Neural Information Processing Systems 36 (2023): 34410-34425.
    
    Implementation copy/pasted and adapted from 
    https://github.com/pmacg/fast-spectral-clustering 
    """
    n = A.shape[ 0 ]
    
    l = min( n_clusters, math.ceil( math.log( n_clusters, 2) ) )
    t = 10 * math.ceil( math.log( n / n_clusters, 2 ) )
    
    #M = g.normalised_signless_laplacian()
    Dhalf = degree_matrix( A, power = -1/2 )
    M = sp.sparse.identity( n ) - Dhalf @ A @ Dhalf
    Y = np.random.normal( size = (n,l) )

    # We know the top eigenvector of the normalised laplacian.
    # It doesn't help with clustering, so we will project our power method to
    # be orthogonal to it.
    
    top_eigvec = np.sqrt( degree_matrix(A) @ np.full( (n,), 1) )
    norm = np.linalg.norm(top_eigvec)
    if norm > 0:
        top_eigvec /= norm

    for _ in range(t):
        Y = M @ Y

        # Project Y to be orthogonal to the top eigenvector
        for i in range(l):
            Y[:, i] -= (top_eigvec.transpose() @ Y[:, i]) * top_eigvec

    kmeans = KMeans(n_clusters = n_clusters, n_init='auto')
    kmeans.fit( Y )
    z = kmeans.labels_ + np.ones( n )
    
    return z.astype(int) 





# =============================================================================
# ADDITIONAL FUNCTIONS
# =============================================================================



def degree_vector( A ):
    return A.sum( axis = 1 ).flatten()


def degree_matrix( A , power=1 ):
    """
    Parameters
    ----------
    A : TYPE
        DESCRIPTION.
    p : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    d = degree_vector( A )
    with np.errstate(divide='ignore'):
        d = d**power
    d[ np.isinf( d ) ] = 0
    #Construct sparse degree matrix
    n = A.shape[0]  #Number of points
    D = sp.sparse.spdiags( d, 0, n, n)

    return D.tocsr()

