#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 2019
@author: Thomas Bonald <bonald@enst.fr>
@author: Bertrand Charpentier <bertrand.charpentier@live.fr>
@author: Quentin Lutz <qlutz@enst.fr>
"""

from typing import Union, Tuple

import numpy as np


def get_labels(dendrogram: np.ndarray, cluster: dict, sort_clusters: bool, return_dendrogram: bool):
    """Returns the labels from clusters."""
    n = dendrogram.shape[0] + 1
    n_clusters = len(cluster)
    clusters = np.array(list(cluster.values()))
    if sort_clusters:
        sizes = np.array([len(nodes) for nodes in clusters])
        index = np.argsort(-sizes)
        clusters = clusters[index]

    labels = np.zeros(n, dtype=int)
    for label, nodes in enumerate(clusters):
        labels[nodes] = label

    if return_dendrogram:
        indices_clusters = np.array(list(cluster.keys()))
        if sort_clusters:
            indices_clusters = indices_clusters[index]
        index_new = np.zeros(2 * n - 1, int)
        index_new[np.array(indices_clusters)] = np.arange(n_clusters)
        index_new[- n_clusters + 1:] = np.arange(n_clusters, 2 * n_clusters - 1)
        dendrogram_new = dendrogram[- n_clusters + 1:].copy()
        dendrogram_new[:, 0] = index_new[dendrogram_new[:, 0].astype(int)]
        dendrogram_new[:, 1] = index_new[dendrogram_new[:, 1].astype(int)]
        return labels, dendrogram_new
    else:
        return labels


def cut_straight(dendrogram: np.ndarray, n_clusters: int = 2, sort_clusters: bool = True,
                 return_dendrogram: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Cut a dendrogram and return the corresponding clustering.

    Parameters
    ----------
    dendrogram:
        Dendrogram
    n_clusters :
        Number of clusters.
    sort_clusters :
        If ``True``,  sorts clusters in decreasing order of size.
    return_dendrogram :
        If ``True``, returns the dendrogram formed by the clusters up to the root.
    Returns
    -------
    labels : np.ndarray
        Cluster of each node.
    dendrogram_new : np.ndarray
        Dendrogram starting from clusters (leaves = clusters).
    """
    if len(dendrogram.shape) != 2 or dendrogram.shape[1] != 4:
        raise ValueError("Check the shape of the dendrogram.")

    n = dendrogram.shape[0] + 1
    if n_clusters < 1 or n_clusters > n:
        raise ValueError("The number of clusters must be between 1 and the number of nodes.")

    cluster = {node: [node] for node in range(n)}
    for t in range(n - n_clusters):
        left = int(dendrogram[t][0])
        right = int(dendrogram[t][1])
        cluster[n + t] = cluster.pop(left) + cluster.pop(right)

    return get_labels(dendrogram, cluster, sort_clusters, return_dendrogram)


def cut_balanced(dendrogram: np.ndarray, max_cluster_size: int = 2, sort_clusters: bool = True,
                 return_dendrogram: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Cuts a dendrogram with a constraint on the cluster size and returns the corresponding clustering.

    Parameters
    ----------
    dendrogram:
        Dendrogram
    max_cluster_size :
        Maximum size of each cluster.
    sort_clusters :
        If ``True``, sort labels in decreasing order of cluster size.
    return_dendrogram :
        If ``True``, returns the dendrogram formed by the clusters up to the root.
    Returns
    -------
    labels : np.ndarray
        Label of each node.
    return_dendrogram :
        If ``True``, returns the dendrogram formed by the clusters up to the root.
    """
    if len(dendrogram.shape) != 2 or dendrogram.shape[1] != 4:
        raise ValueError("Check the shape of the dendrogram.")

    n = dendrogram.shape[0] + 1
    if max_cluster_size < 2 or max_cluster_size > n:
        raise ValueError("The maximum cluster size must be between 2 and the number of nodes.")

    cluster = {node: [node] for node in range(n)}
    for t in range(n - 1):
        left = int(dendrogram[t][0])
        right = int(dendrogram[t][1])
        if left in cluster and right in cluster and len(cluster[left]) + len(cluster[right]) <= max_cluster_size:
            cluster[n + t] = cluster.pop(left) + cluster.pop(right)

    return get_labels(dendrogram, cluster, sort_clusters, return_dendrogram)


