#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 2019
@author: Nathan de Lara <ndelara@enst.fr>
@author: Thomas Bonald <bonald@enst.fr>
"""

from collections import defaultdict
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


def get_index(tree):
    """
    Reindexing of a dendrogram from the leaves

    Parameters
    ----------
    tree:
        The tree to be indexed

    Returns
    -------
    index:
        The index of the root of the given tree
    """
    if type(tree) != list:
        return tree
    else:
        return np.max([get_index(t) for t in tree])


def get_dendrogram(tree, dendrogram=None, index=None, depth=0, size=None, copy_tree=False):
    """
    Reorder a dendrogram in a format compliant with SciPy's visualization from a tree.

    Parameters
    ----------
    tree:
        The initial tree
    dendrogram:
        Intermediary dendrogram for recursive use
    index:
        Intermediary index for recursive use
    depth:
        Current depth for recursive use
    size:
        Current leaf count for recursive use
    copy_tree:
        If True, ensures the passed tree remains unchanged.

    Returns
    -------
    dendrogram:
        The reordered dendrogram
    index:
        The indexing array
    """
    if copy_tree:
        return get_dendrogram(copy.deepcopy(tree))
    else:
        if dendrogram is None:
            dendrogram = []
        if index is None:
            index = get_index(tree)
        if size is None:
            size = defaultdict(lambda: 1)
        if len(tree) > 1:
            lengths = np.array([len(t) for t in tree])
            if np.max(lengths) == 1:
                # merge all
                i = tree.pop()[0]
                j = tree.pop()[0]
                s = size[i] + size[j]
                dendrogram.append([i, j, float(-depth), s])
                index += 1
                while len(tree):
                    s += 1
                    dendrogram.append([index, tree.pop()[0], float(-depth), s])
                    index += 1
                size[index] = s
                tree.append(index)
                return dendrogram, index
            else:
                i = np.argwhere(lengths > 1).ravel()[0]
                dendrogram_, index_ = get_dendrogram(tree[i], None, index, depth + 1, size)
                dendrogram += dendrogram_
                return get_dendrogram(tree, dendrogram, index_, depth, size)
        else:
            return dendrogram, index


def shift_height(dendrogram):
    """
    Shift dendrogram to all non-negative heights for SciPy's visualization.

    Parameters
    ----------
    dendrogram:
        The dendrogram to offset

    Returns
    -------
    The offset dendrogram
    """
    dendrogram = np.array(dendrogram)
    dendrogram[:, 2] += np.max(np.abs(dendrogram[:, 2])) + 1
    return dendrogram.astype(float)
