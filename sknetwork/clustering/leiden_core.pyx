# distutils: language=c++
# cython: language_level=3
from libcpp.set cimport set
from libcpp.vector cimport vector
cimport cython

ctypedef fused int_or_long:
    int
    long

@cython.boundscheck(False)
@cython.wraparound(False)
def optimize_refine_core(int_or_long[:] labels, int_or_long[:] labels_refined, int_or_long[:] indices,
    int_or_long[:] indptr, float[:] data, float[:] out_weights, float[:] in_weights, float[:] out_cluster_weights,
    float[:] in_cluster_weights, float[:] cluster_weights, float[:] self_loops, float resolution):  # pragma: no cover
    """Refine clusters while maximizing modularity.

    Parameters
    ----------
    labels :
        Labels (initial partition).
    labels_refined :
        Refined labels.
    indices :
        CSR format index array of the normalized adjacency matrix.
    indptr :
        CSR format index pointer array of the normalized adjacency matrix.
    data :
        CSR format data array of the normalized adjacency matrix.
    out_weights :
        Out-weights of nodes (sum to 1).
    in_weights :
        In-weights of nodes (sum to 1).
    out_cluster_weights :
        Out-weights of clusters (sum to 1).
    in_cluster_weights :
        In-weights of clusters (sum to 1).
    cluster_weights :
        Weights of clusters (initialized to 0).
    self_loops :
        Weights of self loops.
    resolution :
        Resolution parameter (positive).

    Returns
    -------
    labels_refined :
        Refined labels.
    """
    cdef int_or_long n
    cdef int_or_long label
    cdef int_or_long label_refined
    cdef int_or_long label_target
    cdef int_or_long label_best
    cdef int_or_long i
    cdef int_or_long j
    cdef int_or_long start
    cdef int_or_long end

    cdef float increase = 1
    cdef float delta
    cdef float delta_local
    cdef float delta_best
    cdef float in_weight
    cdef float out_weight

    cdef set[int_or_long] label_set = ()

    n = labels.shape[0]
    while increase:
        increase = 0

        for i in range(n):
            label_set.clear()
            label = labels[i]
            label_refined = labels_refined[i]
            start = indptr[i]
            end = indptr[i+1]

            # neighboring clusters
            for j in range(start, end):
                if labels[indices[j]] == label:
                    label_target = labels_refined[indices[j]]
                    label_set.insert(label_target)
                    cluster_weights[label_target] += data[j]
            label_set.erase(label_refined)

            if not label_set.empty():
                out_weight = out_weights[i]
                in_weight = in_weights[i]

                # node leaving the current cluster
                delta = 2 * (cluster_weights[label_refined] - self_loops[i])
                delta -= resolution * out_weight * (in_cluster_weights[label_refined] - in_weight)
                delta -= resolution * in_weight * (out_cluster_weights[label_refined] - out_weight)

                label_best = label_refined
                delta_best = 0

                for label_target in label_set:
                    delta_local = 2 * cluster_weights[label_target]
                    delta_local -= resolution * out_weight * in_cluster_weights[label_target]
                    delta_local -= resolution * in_weight * out_cluster_weights[label_target]
                    delta_local -= delta
                    if delta_local > 0:
                        # select any label increasing modularity
                        label_best = label_target
                        delta_best = delta_local
                    cluster_weights[label_target] = 0

                if label_best != label_refined:
                    increase += delta_best
                    labels_refined[i] = label_best
                    # update weights
                    out_cluster_weights[label_refined] -= out_weight
                    in_cluster_weights[label_refined] -= in_weight
                    out_cluster_weights[label_best] += out_weight
                    in_cluster_weights[label_best] += in_weight
            cluster_weights[label_refined] = 0

    return labels_refined
