# distutils: language = c++
# cython: language_level=3
# cython: linetrace=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
from libcpp.set cimport set
from libcpp.vector cimport vector
cimport cython

ctypedef fused int_or_long:
    int
    long

@cython.boundscheck(False)
@cython.wraparound(False)
def fit_core(float resolution, float tol, float[:] ou_node_probs, float[:] in_node_probs, float[:] self_loops,
             float[:] data, int_or_long[:] indices, int_or_long[:] indptr):  # pragma: no cover
    """Fit the clusters to the objective function.

    Parameters
    ----------
    resolution :
        Resolution parameter (positive).
    tol :
        Minimum increase in modularity to enter a new optimization pass.
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
    cdef int_or_long n = indptr.shape[0] - 1
    cdef int_or_long increase = 1
    cdef int_or_long cluster
    cdef int_or_long cluster_best
    cdef int_or_long cluster_node
    cdef int_or_long i
    cdef int_or_long j
    cdef int_or_long j1
    cdef int_or_long j2
    cdef int_or_long label

    cdef float increase_total = 0
    cdef float increase_pass
    cdef float delta
    cdef float delta_best
    cdef float delta_exit
    cdef float delta_local
    cdef float node_prob_in
    cdef float node_prob_ou
    cdef float ratio_in
    cdef float ratio_ou

    cdef vector[int_or_long] labels
    cdef vector[float] neighbor_clusters_weights
    cdef vector[float] ou_clusters_weights
    cdef vector[float] in_clusters_weights
    cdef set[int_or_long] unique_clusters = ()

    for i in range(n):
        labels.push_back(i)
        neighbor_clusters_weights.push_back(0.)
        ou_clusters_weights.push_back(ou_node_probs[i])
        in_clusters_weights.push_back(in_node_probs[i])

    while increase == 1:
        increase = 0
        increase_pass = 0

        for i in range(n):
            unique_clusters.clear()
            cluster_node = labels[i]
            j1 = indptr[i]
            j2 = indptr[i + 1]

            for j in range(j1, j2):
                label = labels[indices[j]]
                neighbor_clusters_weights[label] += data[j]
                unique_clusters.insert(label)

            unique_clusters.erase(cluster_node)

            if not unique_clusters.empty():
                node_prob_ou = ou_node_probs[i]
                node_prob_in = in_node_probs[i]
                ratio_ou = resolution * node_prob_ou
                ratio_in = resolution * node_prob_in

                delta_exit = 2 * (neighbor_clusters_weights[cluster_node] - self_loops[i])
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

                    neighbor_clusters_weights[cluster] = 0

                if delta_best > 0:
                    increase_pass += delta_best
                    ou_clusters_weights[cluster_node] -= node_prob_ou
                    in_clusters_weights[cluster_node] -= node_prob_in
                    ou_clusters_weights[cluster_best] += node_prob_ou
                    in_clusters_weights[cluster_best] += node_prob_in
                    labels[i] = cluster_best

            neighbor_clusters_weights[cluster_node] = 0

        increase_total += increase_pass
        if increase_pass > tol:
            increase = 1
    return labels, increase_total
