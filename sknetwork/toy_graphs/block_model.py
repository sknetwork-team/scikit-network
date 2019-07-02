#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul 1, 2019
@author: Quentin Lutz <qlutz@enst.fr>
@author: Nathan de Lara <ndelara@enst.fr>
"""

import numpy as np
from scipy import sparse
from typing import Union, Tuple, Optional
from sknetwork.utils.checks import check_is_proba


def block_model(clusters: Union[np.ndarray, int], shape: Optional[Tuple[int]] = None,
                inner_prob: float = .2, outer_prob: float = .01):
    """
    A block model graph

    Parameters
    ----------
    clusters: Union[np.ndarray, int]
        cluster specifications (array of couples where each line denotes the shape of a cluster
         or an int denoting the number of clusters). If an int is passed, shape must be given and
         the clusters are identical in shape.
    shape: Optional[Tuple[int]]
        The size of the graph to obtain (might be rectangular for a bipartite graph)
    inner_prob: float
        intra-cluster connection probability
    outer_prob: float
        inter-cluster connection probability

    """
    check_is_proba(inner_prob)
    check_is_proba(outer_prob)

    if type(clusters) == int:
        if not shape:
            raise ValueError('Please specify the shape of the matrix when giving a number of clusters.')
        if clusters <= 0:
            raise ValueError('Number of clusters should be positive.')
        n_clusters = clusters
        clusters_cumul = np.zeros((n_clusters + 1, 2), dtype=int)
        row_step, col_step = shape[0] // clusters, shape[1] // clusters
        if row_step == 0 or col_step == 0:
            raise ValueError('Number of clusters is too high given the shape of the matrix.')
        clusters_cumul[:, 0] = np.arange(0, shape[0], row_step)
        clusters_cumul[:, 1] = np.arange(0, shape[1], col_step)
        clusters_cumul[-1, 0] = shape[0]
        clusters_cumul[-1, 1] = shape[1]

    elif type(clusters) == np.ndarray:
        n_clusters = clusters.shape[0]
        clusters_cumul = np.cumsum(clusters, axis=0)
        clusters_cumul = np.insert(clusters_cumul, 0, 0, axis=0)
        if shape:
            if clusters_cumul[-1, 0] != shape[0] or clusters_cumul[-1, 1] != shape[1]:
                raise ValueError('Cluster sizes do not match matrix size.')

    else:
        raise TypeError(
            'Please specify an array of sizes or a number of clusters (along with the shape of the desired matrix).')

    n_rows, n_cols = clusters_cumul[-1, 0], clusters_cumul[-1, 1]
    ground_truth_samples = np.zeros(n_rows, dtype=int)
    ground_truth_features = np.zeros(n_cols, dtype=int)
    mat = sparse.dok_matrix((n_rows, n_cols), dtype=bool)
    for label, row_block in enumerate(range(n_clusters)):
        ground_truth_samples[clusters_cumul[row_block, 0]:clusters_cumul[row_block + 1, 0]] = label
        ground_truth_features[clusters_cumul[row_block, 1]:clusters_cumul[row_block + 1, 1]] = label
        mask = np.full(n_cols, outer_prob)
        mask[clusters_cumul[row_block, 1]:clusters_cumul[row_block + 1, 1]] = inner_prob
        for row in range(clusters_cumul[row_block, 0], clusters_cumul[row_block + 1, 0]):
            mat[row, (np.random.rand(n_cols) < mask)] = True

    return sparse.csr_matrix(mat), ground_truth_features, ground_truth_samples

