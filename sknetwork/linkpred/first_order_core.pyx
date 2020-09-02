# distutils: language = c++
# cython: language_level=3
# cython: linetrace=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
"""
Created on July, 2020
@author: Nathan de Lara <ndelara@enst.fr>
"""
from libc.math cimport log, sqrt
from libcpp.vector cimport vector

ctypedef float (*int2float)(int)
ctypedef float (*vectors2float)(vector[int], vector[int])


cdef float inv(int a):
    """Inverse function"""
    return 1 / a


cdef float inv_log(int a):
    """Inverse of log function"""
    return 1 / log(a)


cdef vector[int] vector_intersection(vector[int] a, vector[int] b):
    """Common elements in two sorted vectors. Each element is assumed unique in each vector."""
    cdef vector[int] intersection
    cdef int e_a, e_b
    cdef int ix_a = 0
    cdef int ix_b = 0
    cdef int size_a = a.size()
    cdef int size_b = b.size()

    while ix_a < size_a and ix_b < size_b:
        e_a = a[ix_a]
        e_b = b[ix_b]

        if e_a < e_b:
            ix_a += 1
        elif e_b < e_a:
            ix_b += 1
        else:
            intersection.push_back(e_a)
            ix_a += 1
            ix_b += 1

    return intersection


cdef float size_intersection(vector[int] a, vector[int] b):
    """Size of the intersection of two vectors"""
    return vector_intersection(a, b).size()


cdef float jaccard(vector[int] a, vector[int] b):
    """Jaccard coefficient"""
    cdef float size_inter = size_intersection(a, b)
    cdef float size_union = a.size() + b.size() - size_inter
    return size_inter / size_union


cdef float salton(vector[int] a, vector[int] b):
    """Salton coefficient"""
    cdef float size_inter = size_intersection(a, b)
    return size_inter / sqrt(a.size() * b.size())


cdef float sorensen(vector[int] a, vector[int] b):
    """Sorensen coefficient"""
    cdef float size_inter = size_intersection(a, b)
    return 2 * size_inter / (a.size() + b.size())


cdef float hub_promoted(vector[int] a, vector[int] b):
    """Hub promoted coefficient"""
    cdef float size_inter = size_intersection(a, b)
    return size_inter / min(a.size(), b.size())


cdef float hub_depressed(vector[int] a, vector[int] b):
    """Hub promoted coefficient"""
    cdef float size_inter = size_intersection(a, b)
    return size_inter / max(a.size(), b.size())


cdef vector[int] neighbors(int[:] indptr, int[:] indices, int node):
    """Neighbors of a given node"""
    cdef int j1 = indptr[node]
    cdef int j2 = indptr[node + 1]
    cdef int j
    cdef vector[int] neigh = ()

    for j in range(j1, j2):
        neigh.push_back(indices[j])

    return neigh


cdef vector[float] predict_node_core(int[:] indptr, int[:] indices, int source, int[:] targets,
                                       vectors2float weight_func):
    """Scores based on global information about common neighbors for a single source.

    Parameters
    ----------
    indptr :
        indptr array of the adjacency matrix
    indices :
        indices array of the adjacency matrix
    source :
        source index
    targets :
        array of target indices
    weight_func :
        scoring function to be used

    Returns
    -------
    scores :
        vector of node pair scores
    """
    cdef int target, i
    cdef int n_targets = targets.shape[0]
    cdef vector[float] preds

    cdef vector[int] neigh_s = neighbors(indptr, indices, source)
    cdef vector[int] neigh_t
    for i in range(n_targets):
        target = targets[i]
        neigh_t = neighbors(indptr, indices, target)
        preds.push_back(weight_func(neigh_s, neigh_t))

    return preds


cdef vector[float] predict_edges_core(int[:] indptr, int[:] indices, int[:, :] edges,
                                       vectors2float weight_func):
    """Scores based on global information about common neighbors for a list of edges.

    Parameters
    ----------
    indptr :
        indptr array of the adjacency matrix
    indices :
        indices array of the adjacency matrix
    edges:
        array of node pairs to be scored
    weight_func :
        scoring function to be used

    Returns
    -------
    scores :
        vector of node pair scores
    """

    cdef vector[float] preds
    cdef int source, target, i

    cdef int n_edges = edges.shape[0]
    for i in range(n_edges):
        source, target = edges[i, 0], edges[i, 1]
        neigh_s = neighbors(indptr, indices, source)
        neigh_t = neighbors(indptr, indices, target)
        preds.push_back(weight_func(neigh_s, neigh_t))

    return preds

def common_neighbors_node_core(int[:] indptr, int[:] indices, int source, int[:] targets):
    """Number of common neighbors"""
    return predict_node_core(indptr, indices, source, targets, size_intersection)

def common_neighbors_edges_core(int[:] indptr, int[:] indices, int[:, :] edges):
    """Number of common neighbors"""
    return predict_edges_core(indptr, indices, edges, size_intersection)

def jaccard_node_core(int[:] indptr, int[:] indices, int source, int[:] targets):
    """Jaccard coefficient of common neighbors"""
    return predict_node_core(indptr, indices, source, targets, jaccard)

def jaccard_edges_core(int[:] indptr, int[:] indices, int[:, :] edges):
    """Number of common neighbors"""
    return predict_edges_core(indptr, indices, edges, jaccard)

def salton_node_core(int[:] indptr, int[:] indices, int source, int[:] targets):
    """Salton coefficient of common neighbors"""
    return predict_node_core(indptr, indices, source, targets, salton)

def salton_edges_core(int[:] indptr, int[:] indices, int[:, :] edges):
    """Salton coefficient of common neighbors"""
    return predict_edges_core(indptr, indices, edges, salton)

def sorensen_node_core(int[:] indptr, int[:] indices, int source, int[:] targets):
    """Sorensen coefficient of common neighbors"""
    return predict_node_core(indptr, indices, source, targets, sorensen)

def sorensen_edges_core(int[:] indptr, int[:] indices, int[:, :] edges):
    """Sorensen coefficient of common neighbors"""
    return predict_edges_core(indptr, indices, edges, sorensen)

def hub_promoted_node_core(int[:] indptr, int[:] indices, int source, int[:] targets):
    """Hub promoted coefficient of common neighbors"""
    return predict_node_core(indptr, indices, source, targets, hub_promoted)

def hub_promoted_edges_core(int[:] indptr, int[:] indices, int[:, :] edges):
    """Hub promoted coefficient of common neighbors"""
    return predict_edges_core(indptr, indices, edges, hub_promoted)

def hub_depressed_node_core(int[:] indptr, int[:] indices, int source, int[:] targets):
    """Hub depressed coefficient of common neighbors"""
    return predict_node_core(indptr, indices, source, targets, hub_depressed)

def hub_depressed_edges_core(int[:] indptr, int[:] indices, int[:, :] edges):
    """Hub depressed coefficient of common neighbors"""
    return predict_edges_core(indptr, indices, edges, hub_depressed)

cdef vector[float] predict_node_weighted_core(int[:] indptr, int[:] indices, int source, int[:] targets,
                                              int2float weight_func):
    """Scores based on the degrees of common neighbors for a single source.

    Parameters
    ----------
    indptr :
        indptr array of the adjacency matrix
    indices :
        indices array of the adjacency matrix
    source :
        source index
    targets :
        array of target indices
    weight_func :
        scoring function to be used

    Returns
    -------
    scores :
        vector of node pair scores
    """
    cdef int target, i, j
    cdef int n_targets = targets.shape[0]
    cdef float weight
    cdef vector[int] intersection
    cdef vector[float] preds

    cdef vector[int] neigh_s = neighbors(indptr, indices, source)
    cdef vector[int] neigh_t
    for i in range(n_targets):
        target = targets[i]
        neigh_t = neighbors(indptr, indices, target)
        intersection = vector_intersection(neigh_s, neigh_t)

        weight = 0
        for j in intersection:
            weight += weight_func(indptr[j+1] - indptr[j])
        preds.push_back(weight)

    return preds


cdef vector[float] predict_edges_weighted_core(int[:] indptr, int[:] indices, int[:, :] edges,
                                               int2float weight_func):
    """Scores based on the degrees of common neighbors for a list of edges.

    Parameters
    ----------
    indptr :
        indptr array of the adjacency matrix
    indices :
        indices array of the adjacency matrix
    edges:
        array of node pairs to be scored
    weight_func :
        scoring function to be used

    Returns
    -------
    scores :
        vector of node pair scores
    """
    cdef vector[float] preds
    cdef int source, target, i
    cdef float weight
    cdef vector[int] intersection

    cdef int n_edges = edges.shape[0]
    for i in range(n_edges):
        source, target = edges[i][0], edges[i][1]
        neigh_s = neighbors(indptr, indices, source)
        neigh_t = neighbors(indptr, indices, target)

        intersection = vector_intersection(neigh_s, neigh_t)

        weight = 0
        for j in intersection:
            weight += weight_func(indptr[j+1] - indptr[j])
        preds.push_back(weight)

    return preds


def adamic_adar_node_core(int[:] indptr, int[:] indices, int source, int[:] targets):
    """Adamic Adar index"""
    return predict_node_weighted_core(indptr, indices, source, targets, inv_log)

def adamic_adar_edges_core(int[:] indptr, int[:] indices, int[:, :] edges):
    """Adamic Adar index"""
    return predict_edges_weighted_core(indptr, indices, edges, inv_log)

def resource_allocation_node_core(int[:] indptr, int[:] indices, int source, int[:] targets):
    """Resource Allocation index"""
    return predict_node_weighted_core(indptr, indices, source, targets, inv)

def resource_allocation_edges_core(int[:] indptr, int[:] indices, int[:, :] edges):
    """Resource Allocation index"""
    return predict_edges_weighted_core(indptr, indices, edges, inv)
