#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 10, 2019
@author: Nathan de Lara <ndelara@enst.fr>
@author: Thomas Bonald <bonald@enst.fr>
"""
import numpy as np


def reindex_labels(labels: np.ndarray, consecutive: bool = True) -> np.ndarray:
    """Reindex clusters in decreasing order of size.

    Parameters
    ----------
    labels :
        label of each node.
    consecutive :
        If ``True``, the set of labels must be from 0 to :math:`k - 1`, where :math:`k` is the number of labels.
        Lead to faster computation.
    Returns
    -------
    new_labels : np.ndarray
        New label of each node.

    Example
    -------
    >>> from sknetwork.clustering import reindex_labels
    >>> labels = np.array([0, 1, 1])
    >>> reindex_labels(labels)
    array([1, 0, 0])
    """
    if not consecutive:
        _, labels = np.unique(labels, return_inverse=True)
    labels_unique, counts = np.unique(labels, return_counts=True)
    sorted_values = labels_unique[np.argsort(-counts)]
    _, index = np.unique(sorted_values, return_index=True)
    labels_ = index[labels.astype(int)]
    return labels_
