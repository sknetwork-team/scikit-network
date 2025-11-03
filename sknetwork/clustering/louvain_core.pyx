# distutils: language=c++
# cython: language_level=3
from libcpp.set cimport set
cimport cython

ctypedef fused int_or_long:
    int
    long long

@cython.boundscheck(False)
@cython.wraparound(False)
def optimize_core(int_or_long[:] labels, int_or_long[:] indices, int_or_long[:] indptr, float[:] data,
    float[:] out_weights, float[:] in_weights, float[:] out_cluster_weights, float[:] in_cluster_weights,
    float[:] cluster_weights, float[:] self_loops, float resolution, float tol_optimization):  # pragma: no cover
    """Find clusters maximizing modularity.

    Parameters
    ----------
    labels :
        Initial labels.
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
    tol_optimization :
        Minimum increase in modularity to enter a new optimization pass.

    Returns
    -------
    labels :
        Labels of nodes.
    increase :
        Increase in modularity.
    """
    cdef int_or_long n
    cdef int_or_long stop = 0
    cdef int_or_long label
    cdef int_or_long label_target
    cdef int_or_long label_best
    cdef int_or_long i
    cdef int_or_long j
    cdef int_or_long start
    cdef int_or_long end

    cdef float increase = 0
    cdef float increase_pass
    cdef float delta
    cdef float delta_local
    cdef float delta_best
    cdef float in_weight
    cdef float out_weight

    cdef set[int_or_long] label_set = ()

    n = labels.shape[0]
    while not stop:
        increase_pass = 0

        for i in range(n):
            label_set.clear()
            label = labels[i]
            start = indptr[i]
            end = indptr[i+1]

            # neighboring clusters
            for j in range(start, end):
                label_target = labels[indices[j]]
                label_set.insert(label_target)
                cluster_weights[label_target] += data[j]
            label_set.erase(label)

            if not label_set.empty():
                out_weight = out_weights[i]
                in_weight = in_weights[i]

                # node leaving the current cluster
                delta = 2 * (cluster_weights[label] - self_loops[i])
                delta -= resolution * out_weight * (in_cluster_weights[label] - in_weight)
                delta -= resolution * in_weight * (out_cluster_weights[label] - out_weight)

                delta_best = 0
                label_best = label

                for label_target in label_set:
                    delta_local = 2 * cluster_weights[label_target]
                    delta_local -= resolution * out_weight * in_cluster_weights[label_target]
                    delta_local -= resolution * in_weight * out_cluster_weights[label_target]
                    delta_local -= delta
                    if delta_local > delta_best:
                        delta_best = delta_local
                        label_best = label_target
                    cluster_weights[label_target] = 0

                if label_best != label:
                    increase_pass += delta_best
                    labels[i] = label_best
                    # update weights
                    out_cluster_weights[label] -= out_weight
                    in_cluster_weights[label] -= in_weight
                    out_cluster_weights[label_best] += out_weight
                    in_cluster_weights[label_best] += in_weight

            cluster_weights[label] = 0

        increase += increase_pass
        stop = increase_pass <= tol_optimization

    return labels, increase
