#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

from typing import Union

import numpy as np
from scipy import sparse

from sknetwork.embedding import SVD, Spectral
from sknetwork.utils.algorithm_base_class import Algorithm
from sknetwork.utils.checks import check_format
from sknetwork.utils.kmeans import KMeans


class SpectralClustering(Algorithm):
    """Pipeline for spectral clustering.

    Parameters
    ----------
    n_clusters:
        Number of desired clusters.
    embedding_dimension:
        Dimension of the embedding on which to apply the clustering.
    clustering_algo:
        Method for clustering, can be ``'kmeans'`` or a custom ``Algorithm``.
    embedding_algo:
        Method for embedding, can be ``'svd'``, ``'spectral'`` or a custom ``Algorithm``.
    l2normalization:
        If ``True``, each row of the embedding is projected onto the L2-sphere before applying the clustering algorithm.

    Attributes
    ----------
    labels_:
        Label of each node.

    """

    def __init__(self, n_clusters: int = 8, embedding_dimension: int = 16,
                 clustering_algo: Union[str, Algorithm] = 'kmeans', embedding_algo: Union[str, Algorithm] = 'svd',
                 l2normalization: bool = True):
        self.n_clusters = n_clusters
        self.embedding_dimension = embedding_dimension

        if clustering_algo == 'kmeans':
            self.clustering_algo = KMeans(n_clusters=n_clusters)
        elif isinstance(clustering_algo, Algorithm):
            self.clustering_algo = clustering_algo
        else:
            raise ValueError('clustering_algo must be either "kmeans" or a custom Algorithm object.')

        if embedding_algo == 'svd':
            self.embedding_algo: Algorithm = SVD(embedding_dimension=embedding_dimension)
        elif embedding_algo == 'spectral':
            self.embedding_algo: Algorithm = Spectral(embedding_dimension=embedding_dimension)
        elif isinstance(embedding_algo, Algorithm):
            self.embedding_algo: Algorithm = embedding_algo
        else:
            raise ValueError('embedding algo must be either "svd", "spectral" or a custom Algorithm object.')

        self.l2normalization = l2normalization

        self.labels_ = None

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'SpectralClustering':
        """Apply embedding method followed by clustering to the graph.

        Parameters
        ----------
        adjacency:
            Adjacency matrix of the graph.

        Returns
        -------
        self: 'SpectralClustering'

        """
        adjacency = check_format(adjacency)
        self.embedding_algo.fit(adjacency)
        if self.l2normalization:
            norm = np.linalg.norm(self.embedding_algo.embedding_, axis=1)
            norm[norm == 0.] = 1
            self.embedding_algo.embedding_ /= norm[:, np.newaxis]
        self.clustering_algo.fit(self.embedding_algo.embedding_)

        self.labels_ = self.clustering_algo.labels_

        return self
