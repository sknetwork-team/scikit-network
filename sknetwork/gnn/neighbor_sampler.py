#!/usr/bin/env python3
# coding: utf-8
"""
@author: Simon Delarue <sdelarue@enst.fr>
"""
from typing import Union

import numpy as np
from scipy import sparse

from sknetwork.utils import get_degrees


class UniformNeighborSampler:
    """Neighbor node sampler.

    Uniformly sample nodes over neighborhood.

    Parameters
    ----------
    sample_size : int
        Size of neighborhood sampled for each node.
    """
    def __init__(self, sample_size: int):
        self.sample_size = sample_size

    def _sample_indexes(self, size: int) -> np.ndarray:
        """Randomly chose indexes without replacement.

        Parameters
        ----------
        size : int
            Highest index available. This index is used if lower than a threshold.

        Returns
        -------
            Array of sampled indexes.
        """
        return np.random.choice(size, size=min(size, self.sample_size), replace=False)

    def __call__(self, adjacency: Union[sparse.csr_matrix, np.ndarray]) -> sparse.csr_matrix:
        """Apply node sampling on each node and return filtered adjacency matrix.

        Parameters
        ----------
        adjacency
            Adjacency matrix of the graph.

        Returns
        -------
            Filtered adjacency matrix using node sampling.
        """
        n_row, _ = adjacency.shape
        sampled_adjacency = adjacency.copy()

        degrees = get_degrees(adjacency)
        neighbor_samples = list(map(self._sample_indexes, degrees))

        for i, neighbors in enumerate(neighbor_samples):
            sampled_adjacency.data[sampled_adjacency.indptr[i]:sampled_adjacency.indptr[i + 1]] = np.zeros(degrees[i])
            sampled_adjacency.data[sampled_adjacency.indptr[i]:sampled_adjacency.indptr[i + 1]][neighbors] = 1

        sampled_adjacency.eliminate_zeros()

        return sampled_adjacency
