#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 10, 2019
@author: Nathan de Lara <ndelara@enst.fr>
@author: Thomas Bonald <bonald@enst.fr>
"""

import numpy as np


def reindex_clusters(labels: np.ndarray, assume_range: bool = True) -> np.ndarray:
    """Reindex clusters in decreasing order of size.

    Parameters
    ----------
    labels:
        label of each node.
    assume_range:
        If True, the labels are assumed to be between 0 and k-1, this leads to faster computation.
    Returns
    -------
    new_labels: np.ndarray
        new label of each node.
    """
    if not assume_range:
        _, labels = np.unique(labels, return_inverse=True)
    unique_values, counts = np.unique(labels, return_counts=True)
    sorted_values = unique_values[np.argsort(-counts)]
    _, new_index = np.unique(sorted_values, return_index=True)
    new_labels = new_index[labels.astype(int)]
    return new_labels
