#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 2019
@author: Thomas Bonald <bonald@enst.fr>
@author: Bertrand Charpentier <bertrand.charpentier@live.fr>
@author: Quentin Lutz <qlutz@enst.fr>
"""

import copy
from collections import defaultdict
from typing import Optional, Union, Tuple

import numpy as np

from sknetwork.utils.check import check_n_clusters, check_dendrogram


def reorder_dendrogram(dendrogram: np.ndarray) -> np.ndarray:
    """Reorder the dendrogram in non-decreasing order of height."""
    n = dendrogram.shape[0] + 1
    order = np.zeros((2, n - 1), float)
    order[0] = np.max(dendrogram[:, :2], axis=1)
    order[1] = dendrogram[:, 2]
    index = np.lexsort(order)
    dendrogram_new = dendrogram[index]
    index_new = np.arange(2 * n - 1)
    index_new[n + index] = np.arange(n, 2 * n - 1)
    dendrogram_new[:, 0] = index_new[dendrogram_new[:, 0].astype(int)]
    dendrogram_new[:, 1] = index_new[dendrogram_new[:, 1].astype(int)]
    return dendrogram_new


def get_labels(dendrogram: np.ndarray, cluster: dict, sort_clusters: bool, return_dendrogram: bool):
    """Returns the labels from clusters."""
    n = len(dendrogram) + 1
    clusters = list(cluster.values())
    if sort_clusters:
        sizes = np.array([len(nodes) for nodes in clusters])
        index = np.argsort(-sizes)
        clusters = [clusters[i] for i in index]

    labels = np.zeros(n, dtype=int)
    for label, nodes in enumerate(clusters):
        labels[nodes] = label

    if return_dendrogram:
        cluster_index = {i: label for i, label in enumerate(labels)}
        cluster_size = {i: len(cluster) for i, cluster in enumerate(clusters)}
        dendrogram_new = []
        current_cluster = len(labels)
        current_cluster_new = len(clusters)
        for i, j, height, _ in dendrogram:
            i_new = cluster_index.pop(int(i))
            j_new = cluster_index.pop(int(j))
            if i_new != j_new:
                size = cluster_size.pop(i_new) + cluster_size.pop(j_new)
                cluster_size[current_cluster_new] = size
                cluster_index[current_cluster] = current_cluster_new
                dendrogram_new.append([i_new, j_new, height, size])
                current_cluster_new += 1
            else:
                cluster_index[current_cluster] = i_new
            current_cluster += 1
        dendrogram_new = np.array(dendrogram_new)
        return labels, dendrogram_new
    else:
        return labels


def cut_straight(dendrogram: np.ndarray, n_clusters: Optional[int] = None, threshold: Optional[float] = None,
                 sort_clusters: bool = True, return_dendrogram: bool = False) \
                -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Cut a dendrogram and return the corresponding clustering.

    Parameters
    ----------
    dendrogram : np.ndarray
        Dendrogram.
    n_clusters : int
        Number of clusters (optional).
        The number of clusters can be larger than n_clusters in case of equal heights in the dendrogram.
    threshold : float
        Threshold on height (optional).
        If both n_clusters and threshold are ``None``, n_clusters is set to 2.
    sort_clusters : bool
        If ``True``,  sorts clusters in decreasing order of size.
    return_dendrogram : bool
        If ``True``, returns the dendrogram formed by the clusters up to the root.
    Returns
    -------
    labels : np.ndarray
        Cluster of each node.
    dendrogram_aggregate : np.ndarray
        Dendrogram starting from clusters (leaves = clusters).

    Example
    -------
    >>> from sknetwork.hierarchy import cut_straight
    >>> dendrogram = np.array([[0, 1, 0, 2], [2, 3, 1, 3]])
    >>> cut_straight(dendrogram)
    array([0, 0, 1])
    """
    check_dendrogram(dendrogram)
    n = dendrogram.shape[0] + 1

    if return_dendrogram:
        height = dendrogram[:, 2]
        if not np.all(height[:-1] <= height[1:]):
            dendrogram = reorder_dendrogram(dendrogram)

    cluster = {i: [i] for i in range(n)}
    if n_clusters is None:
        if threshold is None:
            n_clusters = 2
        else:
            n_clusters = n
    else:
        check_n_clusters(n_clusters, n, n_min=1)
    cut = np.sort(dendrogram[:, 2])[n - n_clusters]
    if threshold is not None:
        cut = max(cut, threshold)
    for t in range(n - 1):
        i = int(dendrogram[t][0])
        j = int(dendrogram[t][1])
        if dendrogram[t][2] < cut and i in cluster and j in cluster:
            cluster[n + t] = cluster.pop(i) + cluster.pop(j)

    return get_labels(dendrogram, cluster, sort_clusters, return_dendrogram)


def cut_balanced(dendrogram: np.ndarray, max_cluster_size: int = 20, sort_clusters: bool = True,
                 return_dendrogram: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Cuts a dendrogram with a constraint on the cluster size and returns the corresponding clustering.

    Parameters
    ----------
    dendrogram : np.ndarray
        Dendrogram
    max_cluster_size : int
        Maximum size of each cluster.
    sort_clusters : bool
        If ``True``, sort labels in decreasing order of cluster size.
    return_dendrogram : bool
        If ``True``, returns the dendrogram formed by the clusters up to the root.
    Returns
    -------
    labels : np.ndarray
        Label of each node.
    dendrogram_aggregate : np.ndarray
        Dendrogram starting from clusters (leaves = clusters).

    Example
    -------
    >>> from sknetwork.hierarchy import cut_balanced
    >>> dendrogram = np.array([[0, 1, 0, 2], [2, 3, 1, 3]])
    >>> cut_balanced(dendrogram, 2)
    array([0, 0, 1])
    """
    check_dendrogram(dendrogram)
    n = dendrogram.shape[0] + 1
    if max_cluster_size < 2 or max_cluster_size > n:
        raise ValueError("The maximum cluster size must be between 2 and the number of nodes.")

    cluster = {i: [i] for i in range(n)}
    for t in range(n - 1):
        i = int(dendrogram[t][0])
        j = int(dendrogram[t][1])
        if i in cluster and j in cluster and len(cluster[i]) + len(cluster[j]) <= max_cluster_size:
            cluster[n + t] = cluster.pop(i) + cluster.pop(j)

    return get_labels(dendrogram, cluster, sort_clusters, return_dendrogram)


def aggregate_dendrogram(dendrogram: np.ndarray, n_clusters: int = 2, return_counts: bool = False) \
                        -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Aggregate a dendrogram in order to get a certain number of leaves.
    The leaves in the output dendrogram correspond to subtrees in the input one.

    Parameters
    ----------
    dendrogram : np.ndarray
        The input to aggregate.
    n_clusters : int
        Number of clusters (or leaves) to keep.
    return_counts :  bool
        If ``True``, returns an array of counts corresponding to the sizes of the merged subtrees.
        The sum of the counts is equal to the number of samples in the input dendrogram.

    Returns
    -------
    new_dendrogram : np.ndarray
        Aggregated dendrogram. The nodes are reindexed from 0.
    counts : np.ndarray
        Size of the subtrees corresponding to each leaf in new_dendrogram.
    """
    n_nodes: int = dendrogram.shape[0] + 1
    check_n_clusters(n_clusters, n_nodes, n_min=1)

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
    """Reindex a dendrogram from the leaves

    Parameters
    ----------
    tree :
        The tree to be indexed

    Returns
    -------
    index :
        The index of the root of the given tree
    """
    if type(tree) != list:
        return tree
    else:
        return np.max([get_index(t) for t in tree])


def get_dendrogram(tree, dendrogram=None, index=None, depth=0, size=None, copy_tree=False):
    """Get dendrogram from tree.

    Parameters
    ----------
    tree :
        The initial tree
    dendrogram :
        Intermediary dendrogram for recursive use
    index :
        Intermediary index for recursive use
    depth :
        Current depth for recursive use
    size :
        Current leaf count for recursive use
    copy_tree :
        If ``True``, ensure the passed tree remains unchanged.

    Returns
    -------
    dendrogram`:
        The reordered dendrogram
    index :
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


def split_dendrogram(dendrogram: np.ndarray, shape: tuple):
    """Split the dendrogram of a bipartite graph into 2 dendrograms, one for each part.

    Parameters
    ----------
    dendrogram :
        Dendrogram of the bipartite graph.
    shape :
        Shape of the biadjacency matrix.
    Returns
    -------
    dendrogram_row :
        Dendrogram for the rows.
    dendrogram_col :
        Dendrogram for the columns.
    """
    n1, n2 = shape
    dendrogram_row = []
    dendrogram_col = []
    id_row_new = n1
    id_col_new = n2
    size_row = {i: 1 for i in range(n1)}
    size_col = {i + n1: 1 for i in range(n2)}
    id_row = {i: i for i in range(n1)}
    id_col = {i + n1: i for i in range(n2)}

    for t in range(n1 + n2 - 1):
        i = dendrogram[t, 0]
        j = dendrogram[t, 1]

        if i in id_row and j in id_row:
            size_row[n1 + n2 + t] = size_row.pop(i) + size_row.pop(j)
            id_row[n1 + n2 + t] = id_row_new
            dendrogram_row.append([id_row.pop(i), id_row.pop(j), dendrogram[t, 2], size_row[n1 + n2 + t]])
            id_row_new += 1
        elif i in id_row:
            size_row[n1 + n2 + t] = size_row.pop(i)
            id_row[n1 + n2 + t] = id_row.pop(i)
        elif j in id_row:
            size_row[n1 + n2 + t] = size_row.pop(j)
            id_row[n1 + n2 + t] = id_row.pop(j)

        if i in id_col and j in id_col:
            size_col[n1 + n2 + t] = size_col.pop(i) + size_col.pop(j)
            id_col[n1 + n2 + t] = id_col_new
            dendrogram_col.append([id_col.pop(i), id_col.pop(j), dendrogram[t, 2], size_col[n1 + n2 + t]])
            id_col_new += 1
        elif i in id_col:
            size_col[n1 + n2 + t] = size_col.pop(i)
            id_col[n1 + n2 + t] = id_col.pop(i)
        elif j in id_col:
            size_col[n1 + n2 + t] = size_col.pop(j)
            id_col[n1 + n2 + t] = id_col.pop(j)

    return np.array(dendrogram_row), np.array(dendrogram_col)
