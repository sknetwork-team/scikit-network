#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

from typing import Union

import numpy as np
from scipy import sparse

from sknetwork.embedding import Spectral
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
    l2normalization:
        If ``True``, each row of the embedding is projected onto the L2-sphere before applying the clustering algorithm.
    co_clustering:
        If ``True``, jointly cluster the rows and columns of bipartite graphs. Otherwise, only cluster the rows.

    Attributes
    ----------
    labels_:
        Labels the rows.
    col_labels_:
        Labels of the columns, on valid if ``co_clustering=True``.

    """

    def __init__(self, n_clusters: int = 8, embedding_dimension: int = 16, l2normalization: bool = True,
                 co_clustering: bool = False):
        self.n_clusters = n_clusters
        self.embedding_dimension = embedding_dimension
        self.l2normalization = l2normalization
        self.co_clustering = co_clustering

        self.labels_ = None
        self.col_labels_ = None

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
        n1, n2 = adjacency.shape

        spectral = Spectral(self.embedding_dimension).fit(adjacency, force_biadjacency=self.co_clustering)
        kmeans = KMeans(self.n_clusters)

        if self.co_clustering:
            embedding = np.vstack((spectral.embedding_, spectral.col_embedding_))
        else:
            embedding = spectral.embedding_

        if self.l2normalization:
            norm = np.linalg.norm(embedding, axis=1)
            norm[norm == 0.] = 1
            embedding /= norm[:, np.newaxis]

        kmeans.fit(embedding)

        if self.co_clustering:
            self.labels_ = kmeans.labels_[:n1]
            self.col_labels_ = kmeans.labels_[n1:]
        else:
            self.labels_ = kmeans.labels_

        return self
