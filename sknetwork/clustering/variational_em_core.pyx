# distutils: language = c++
# cython: language_level=3
"""
Created on Sep 2020
@author: Clément Bonet <cbonet@enst.fr>
"""
cimport cython
from cython.parallel import prange

import numpy as np
cimport numpy as np

from libc.math cimport log

cdef float eps = np.finfo(float).eps

@cython.boundscheck(False)
@cython.wraparound(False)
def likelihood_core(int[:] indptr,int[:] indices, float[:,:] membership_probs, float[:] cluster_mean_probs,
		float[:,:] cluster_transition_probs) -> float:
    """Compute the approximated likelihood

    Parameters
    ----------
    indptr:
        Index pointer array of the adjacency matrix of the graph (np.ndarray because of numba, could probably be
        modified in csr_matrix).
    indices:
        Indices array of the adjacency matrix of the graph (np.ndarray because of numba, could probably be
        modified in csr_matrix).
    membership_probs:
        Membership matrix given as a probability over clusters.
    cluster_mean_probs:
        Average value of cluster probability over nodes.
    cluster_transition_probs:
        Probabilities of transition from one cluster to another in one hop.

    Returns
    -------
    part_likelihood: float
    """
    cdef int n = indptr.shape[0] - 1
    cdef int n_clusters = membership_probs.shape[1]

    cdef float cpt = 0
    cdef int i
    cdef int j
    cdef int cluster_1
    cdef int cluster_2
    cdef float logb
    cdef int ind1, ind2, ind

    for i in prange(n, nogil=True, schedule='guided'):
        for cluster_1 in prange(n_clusters, schedule='guided'):
            for cluster_2 in prange(n_clusters, schedule='guided'):
                for j in range(n):
                    if i != j:
                        logb = log(1 - cluster_transition_probs[cluster_1, cluster_2])
                        cpt += membership_probs[i, cluster_1] * membership_probs[j, cluster_2] * logb

                ind1 = indptr[i]
                ind2 = indptr[i + 1]
                for ind in range(ind1, ind2):
                    j = indices[ind]
                    if j != i:
                        logb = log(cluster_transition_probs[cluster_1, cluster_2]) \
                                - log(1 - cluster_transition_probs[cluster_1, cluster_2])
                        cpt += membership_probs[i, cluster_1] * membership_probs[j, cluster_2] * logb

    return cpt


@cython.boundscheck(False)
@cython.wraparound(False)
def variational_step_core(int[:] indptr, int[:] indices, float[:,:] membership_probs,
             float[:] cluster_mean_probs, float[:,:] cluster_transition_probs):
    """Apply the variational step:
    - update membership_probas

    Parameters
    ----------
    indptr:
        Index pointer array of the adjacency matrix of the graph (np.ndarray because of numba, could probably be
        modified in csr_matrix).
    indices:
        Indices array of the adjacency matrix of the graph (np.ndarray because of numba, could probably be
        modified in csr_matrix).
    membership_probs:
        Membership matrix given as a probability over clusters.
    cluster_mean_probs:
        Average value of cluster probability over nodes.
    cluster_transition_probs:
        Probabilities of transition from one cluster to another in one hop.

    Returns
    -------
    log_membership_probs:
        Updated membership matrix given as a log probability over clusters.
    """
    cdef int n = indptr.shape[0] - 1
    cdef int n_clusters = membership_probs.shape[1]
    cdef float[:,:] log_membership_prob = np.log(np.maximum(membership_probs, eps))
    cdef int i
    cdef int j
    cdef int cluster_1
    cdef int cluster_2
    cdef int ind1, ind2, ind

    for i in range(n):
        for cluster in range(n_clusters):
            log_membership_prob[i, cluster] = log(cluster_mean_probs[cluster])

        for cluster_1 in prange(n_clusters, nogil=True, schedule='guided'):
            for cluster_2 in prange(n_clusters, schedule='guided'):
                for j in range(n):
                    if j != i:
                        log_membership_prob[i, cluster_1] += \
                            membership_probs[j, cluster_2] * log(1 - cluster_transition_probs[cluster_1, cluster_2])

                ind1 = indptr[i]
                ind2 = indptr[i + 1]
                for ind in range(ind1, ind2):
                    j = indices[ind]
                    if j != i:
                        log_membership_prob[i, cluster_1] += \
                            membership_probs[j, cluster_2] * \
                            (log(cluster_transition_probs[cluster_1, cluster_2])
                            - log(1 - cluster_transition_probs[cluster_1, cluster_2]))

    return log_membership_prob
