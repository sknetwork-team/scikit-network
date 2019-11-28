#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

from typing import Union, Tuple

import numpy as np


def aggregate_dendrogram(dendrogram: np.ndarray, n_clusters: int = 2, return_counts: bool = False) \
                        -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Aggregate a dendrogram in order to get a certain number of leaves.
    The leaves in the output dendrogram correspond to subtrees in the input one.

    Parameters
    ----------
    dendrogram:
        The input to aggregate.
    n_clusters:
        Number of clusters (or leaves) to keep.
    return_counts
        If ``True``, returns an array of counts corresponding to the sizes of the merged subtrees.
        The sum of the counts is equal to the number of samples in the input dendrogram.

    Returns
    -------
    new_dendrogram:
        Aggregated dendrogram. The nodes are reindexed from 0.
    counts:
        Size of the subtrees corresponding to each leaf in new_dendrogram.
    """
    n_nodes: int = dendrogram.shape[0] + 1
    if n_clusters < 1 or n_clusters > n_nodes:
        raise ValueError("The number of clusters must be between 1 and the number of nodes.")

    new_dendrogram = dendrogram[n_nodes - n_clusters:].copy()
    node_indices = np.array(sorted(set(new_dendrogram[:, 0]).union(set(new_dendrogram[:, 1]))))
    new_index = {ix: i for i, ix in enumerate(node_indices)}

    for j in range(2):
        for i in range(new_dendrogram.shape[0]):
            new_dendrogram[i, j] = new_index[new_dendrogram[i, j]]

    if return_counts:
        leaves = node_indices[:n_clusters].astype(int)
        leaves_indices = leaves - n_nodes
        counts = dendrogram[leaves_indices, 3]

        return new_dendrogram, counts.astype(int)
    else:
        return new_dendrogram
