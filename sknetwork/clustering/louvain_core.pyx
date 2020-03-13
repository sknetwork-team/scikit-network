import numpy as np
cimport numpy as np

from libcpp.set cimport set
cimport cython

ctypedef np.int_t int_type_t
ctypedef np.float_t float_type_t

@cython.boundscheck(False)
@cython.wraparound(False)
def fit_core(float_type_t resolution, float_type_t tol, int_type_t n_nodes, np.float_t[:] ou_node_probs,
             np.float_t[:] in_node_probs, np.float_t[:] self_loops, np.float_t[:] data, int[:] indices,
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
    ou_node_probs :
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
    cdef np.float_t[:] neighbor_clusters_weights = ou_node_probs.copy()
    cdef np.float_t[:] ou_clusters_weights = ou_node_probs.copy()
    cdef np.float_t[:] in_clusters_weights = in_node_probs.copy()
    cdef int[:] neighbors
    cdef np.float_t[:] weights
    cdef set[int] unique_clusters = ()

    cdef float increase_pass
    cdef float increase_total = 0
    cdef float delta
    cdef float delta_best
    cdef float delta_exit
    cdef float delta_local
    cdef float node_prob_in
    cdef float node_prob_ou
    cdef float ratio_in
    cdef float ratio_ou

    cdef int cluster
    cdef int cluster_best
    cdef int cluster_node
    cdef int i
    cdef int label
    cdef int n_neighbors
    cdef int node

    while increase == 1:
        increase = 0
        increase_pass = 0

        for node in range(n_nodes):
            has_candidates = 0
            cluster_node = labels[node]
            neighbors = indices[indptr[node]:indptr[node + 1]]
            n_neighbors = neighbors.shape[0]
            weights = data[indptr[node]:indptr[node + 1]]


            neighbor_clusters_weights[:] = 0

            unique_clusters.clear()
            for i in range(n_neighbors):
                label = labels[neighbors[i]]
                neighbor_clusters_weights[label] += weights[i]
                unique_clusters.insert(label)

            unique_clusters.erase(cluster_node)

            if not unique_clusters.empty():
                node_prob_ou = ou_node_probs[node]
                node_prob_in = in_node_probs[node]
                ratio_ou = resolution * node_prob_ou
                ratio_in = resolution * node_prob_in

                delta_exit = 2 * (neighbor_clusters_weights[cluster_node] - self_loops[node])
                delta_exit -= ratio_ou * (in_clusters_weights[cluster_node] - node_prob_in)
                delta_exit -= ratio_in * (ou_clusters_weights[cluster_node] - node_prob_ou)

                delta_best = 0
                cluster_best = cluster_node

                for cluster in unique_clusters:
                    delta = 2 * neighbor_clusters_weights[cluster]
                    delta -= ratio_ou * in_clusters_weights[cluster]
                    delta -= ratio_in * ou_clusters_weights[cluster]

                    delta_local = delta - delta_exit
                    if delta_local > delta_best:
                        delta_best = delta_local
                        cluster_best = cluster

                if delta_best > 0:
                    increase_pass += delta_best
                    ou_clusters_weights[cluster_node] -= node_prob_ou
                    in_clusters_weights[cluster_node] -= node_prob_in
                    ou_clusters_weights[cluster_best] += node_prob_ou
                    in_clusters_weights[cluster_best] += node_prob_in
                    labels[node] = cluster_best

        increase_total += increase_pass
        if increase_pass > tol:
            increase = 1
    return labels, increase_total
