#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 2019
@author: Thomas Bonald <bonald@enst.fr>
@author: Bertrand Charpentier <bertrand.charpentier@live.fr>
@author: Quentin Lutz <qlutz@enst.fr>
"""

import numpy as np


def straight_cut(dendrogram: np.ndarray, n_clusters: int = 2, sorted_clusters: bool = True) -> np.ndarray:
    """
    Extract the clustering with given number of clusters from the dendrogram.

    Parameters
    ----------
    dendrogram:
        Dendrogram
    n_clusters :
        Number of cluster.
    sorted_clusters :
        If True, sort labels in decreasing order of cluster size.

    Returns
    -------
    labels : np.ndarray
        Cluster index of each node.
    """
    n_nodes = dendrogram.shape[0] + 1
    if n_clusters < 1 or n_clusters > n_nodes:
        raise ValueError("The number of clusters must be between 1 and the number of nodes.")

    labels = np.zeros(n_nodes, dtype=int)
    cluster = {node: [node] for node in range(n_nodes)}
    for t in range(n_nodes - n_clusters):
        left = int(dendrogram[t][0])
        right = int(dendrogram[t][1])
        cluster[n_nodes + t] = cluster.pop(left) + cluster.pop(right)

    clusters = list(cluster.values())
    if sorted_clusters:
        clusters = sorted(clusters, key=len, reverse=True)
    for label, index in enumerate(clusters):
        labels[index] = label
    return labels


def balanced_cut(dendrogram: np.ndarray, max_cluster_size: int = 2, sorted_clusters: bool = True) -> np.ndarray:
    """
    Extract the clustering from the dendrogram with some maximum size for each cluster.

    Parameters
    ----------
    dendrogram:
        Dendrogram
    max_cluster_size :
        Maximum size of each cluster.
    sorted_clusters :
        If True, sort labels in decreasing order of cluster size.

    Returns
    -------
    labels : np.ndarray
        Cluster index of each node.
    """
    n_nodes = dendrogram.shape[0] + 1
    if max_cluster_size < 2 or max_cluster_size > n_nodes:
        raise ValueError("The maximum cluster size must be between 2 and the number of nodes.")

    labels = np.zeros(n_nodes, dtype=int)
    cluster = {node: [node] for node in range(n_nodes)}
    for t in range(n_nodes - 1):
        left = int(dendrogram[t][0])
        right = int(dendrogram[t][1])
        if left in cluster and right in cluster and len(cluster[left]) + len(cluster[right]) <= max_cluster_size:
            cluster[n_nodes + t] = cluster.pop(left) + cluster.pop(right)
            
    clusters = list(cluster.values())
    if sorted_clusters:
        clusters = sorted(clusters, key=len, reverse=True)
    for label, index in enumerate(clusters):
        labels[index] = label
    return labels

