cimport numpy as np
import numpy as np

cimport cython

ctypedef np.int_t int_type_t
ctypedef np.float_t float_type_t

@cython.boundscheck(False)
@cython.wraparound(False)
def fit_core(float_type_t resolution,float_type_t tol,int_type_t n_nodes,np.float_t[:] out_node_probs,
             np.float_t[:] in_node_probs,np.float_t[:] self_loops,np.float_t[:] data,int[:] indices,
             int[:] indptr):  # pragma: no cover
    """
    Fit the clusters to the objective function.

    Parameters
    ----------
    resolution :
        Resolution parameter (positive).
    tol :
        Minimum increase in modularity to enter a new optimization pass.
    n_nodes :
        Number of nodes.
    out_node_probs :
        Distribution of node weights based on their out-edges (sums to 1).
    in_node_probs :
        Distribution of node weights based on their in-edges (sums to 1).
    self_loops :
        Weights of self loops.
    data :
        CSR format data array of the normalized adjacency matrix.
    indices :
        CSR format index array of the normalized adjacency matrix.
    indptr :
        CSR format index pointer array of the normalized adjacency matrix.

    Returns
    -------
    labels :
        Cluster index of each node.
    total_increase :
        Score of the clustering (total increase in modularity).
    """
    cdef int increase = 1
    cdef int has_candidates = 0

    cdef int[:] labels = np.arange(n_nodes, dtype=np.intc)
    cdef np.float_t[:] neighbor_clusters_weights = np.zeros(n_nodes, dtype=np.float)
    cdef np.float_t[:] out_clusters_weights = out_node_probs.copy()
    cdef np.float_t[:] in_clusters_weights = in_node_probs.copy()
    cdef int[:] neighbors
    cdef np.float_t[:] weights
    cdef int[:] labels_neighbors
    cdef int[:] unique_clusters

    cdef float total_increase = 0
    cdef float pass_increase
    cdef float exit_delta
    cdef float best_delta
    cdef float local_delta
    cdef float delta
    cdef float out_ratio
    cdef float in_ratio

    cdef int node_cluster
    cdef int best_cluster
    cdef int cluster
    cdef int node
    cdef int i
    cdef int n_neighbors

    while increase == 1:
        increase = 0
        pass_increase = 0

        for node in range(n_nodes):
            has_candidates = 0
            node_cluster = labels[node]
            neighbors = indices[indptr[node]:indptr[node + 1]]
            weights = data[indptr[node]:indptr[node + 1]]
            n_neighbors = len(neighbors)
            labels_neighbors = np.zeros(n_neighbors, dtype=np.intc)

            for i in range(n_nodes):
                neighbor_clusters_weights[i] = 0
            for i in range(len(neighbors)):
                if labels[neighbors[i]] != node_cluster:
                    has_candidates = 1
                neighbor_clusters_weights[labels[neighbors[i]]] += weights[i]
                labels_neighbors[i] = labels[neighbors[i]]

            unique_clusters = np.unique(labels_neighbors)


            if has_candidates == 1:
                out_ratio = resolution * out_node_probs[node]
                in_ratio = resolution * in_node_probs[node]
                exit_delta = 2 * (neighbor_clusters_weights[node_cluster] - self_loops[node])
                exit_delta -= out_ratio * (in_clusters_weights[node_cluster] - in_node_probs[node])
                exit_delta -= in_ratio * (out_clusters_weights[node_cluster] - out_node_probs[node])

                best_delta = 0
                best_cluster = node_cluster
                neighbor_clusters_weights[node_cluster] = 0


                for i in range(len(unique_clusters)):
                    cluster = unique_clusters[i]
                    if cluster != node_cluster:
                        delta = 2 * neighbor_clusters_weights[cluster]
                        delta -= out_ratio * in_clusters_weights[cluster]
                        delta -= in_ratio * out_clusters_weights[cluster]

                        local_delta = delta - exit_delta
                        if local_delta > best_delta:
                            best_delta = local_delta
                            best_cluster = cluster

                if best_delta > 0:
                    pass_increase += best_delta
                    out_clusters_weights[node_cluster] -= out_node_probs[node]
                    in_clusters_weights[node_cluster] -= in_node_probs[node]
                    out_clusters_weights[best_cluster] += out_node_probs[node]
                    in_clusters_weights[best_cluster] += in_node_probs[node]
                    labels[node] = best_cluster

        total_increase += pass_increase
        if pass_increase > tol:
            increase = 1
    return labels, total_increase
