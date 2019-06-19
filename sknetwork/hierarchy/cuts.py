#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 2019
@author: Thomas Bonald <bonald@enst.fr>
@author: Bertrand Charpentier <bertrand.charpentier@live.fr>
@author: Quentin Lutz <qlutz@enst.fr>
"""

from sknetwork.utils.checks import *


def cut(dendrogram: np.ndarray, n_clusters: int = 2, sorted_clusters: bool = False) -> np.ndarray:
    """
    Extract the clustering with given number of clusters from the dendrogram.

    Parameters
    ----------
    dendrogram:
        Dendrogram
    n_clusters :
        Number of clusters.
    sorted_clusters :
        If True, sort labels in decreasing order of cluster size.

    Returns
    -------
    labels : np.ndarray
        Cluster index of each node.
    """
    n_nodes = np.shape(dendrogram)[0] + 1
    if n_clusters < 1 or n_clusters > n_nodes:
        raise ValueError("The number of clusters must be between 1 and the number of nodes.")

    labels = np.zeros(n_nodes, dtype=int)
    clusters = {node: [node] for node in range(n_nodes)}
    for t in range(n_nodes - n_clusters):
        clusters[n_nodes + t] = clusters.pop(int(dendrogram[t][0])) + \
                                clusters.pop(int(dendrogram[t][1]))
    clusters = list(clusters.values())
    if sorted_clusters:
        clusters = sorted(clusters, key=len, reverse=True)
    for label, cluster in enumerate(clusters):
        labels[cluster] = label
    return labels
