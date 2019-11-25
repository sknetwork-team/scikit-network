#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

from typing import Union

import numpy as np
from scipy import sparse

from sknetwork.clustering.base import BaseClustering
from sknetwork.embedding import BiSpectral, Spectral
from sknetwork.utils.checks import check_format, is_symmetric
from sknetwork.utils.kmeans import KMeans


class SpectralClustering(BaseClustering):
    """Pipeline for spectral clustering.

    Parameters
    ----------
    n_clusters:
        Number of desired clusters.
    embedding_dimension:
        Dimension of the embedding on which to apply the clustering.
    l2normalization:
        If ``True``, each row of the embedding is projected onto the L2-sphere before applying the clustering algorithm.

    Attributes
    ----------
    labels_:
        Labels of the rows.

    """

    def __init__(self, n_clusters: int = 8, embedding_dimension: int = 16, l2normalization: bool = True):
        super(SpectralClustering, self).__init__()

        self.n_clusters = n_clusters
        self.embedding_dimension = embedding_dimension
        self.l2normalization = l2normalization

    # noinspection DuplicatedCode
    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'SpectralClustering':
        """Apply embedding method followed by clustering to the graph.

        Parameters
        ----------
        adjacency:
            Adjacency matrix of the graph.

        Returns
        -------
        self: :class:`SpectralClustering`

        """
        adjacency = check_format(adjacency)
        if not is_symmetric(adjacency):
            raise ValueError('The adjacency is not symmetric.')

        spectral = Spectral(self.embedding_dimension).fit(adjacency)
        embedding = spectral.embedding_

        if self.l2normalization:
            norm = np.linalg.norm(embedding, axis=1)
            norm[norm == 0.] = 1
            embedding /= norm[:, np.newaxis]

        kmeans = KMeans(self.n_clusters)
        kmeans.fit(embedding)

        self.labels_ = kmeans.labels_

        return self


class BiSpectralClustering(SpectralClustering):
    """Pipeline for spectral biclustering.

    Parameters
    ----------
    n_clusters:
        Number of desired clusters.
    embedding_dimension:
        Dimension of the embedding on which to apply the clustering.
    l2normalization:
        If ``True``, each row of the embedding is projected onto the L2-sphere before applying the clustering algorithm.
    co_clustering:
        If ``True``, jointly clusters rows and columns of the biadjacency matrix.
        Otherwise, only cluster the rows.

    Attributes
    ----------
    row_labels_: np.ndarray
        Labels of the rows.
    col_labels_: np.ndarray
        Labels of the columns. Only valid if ``co_clustering=True``.
    labels_: np.ndarray
        Labels of rows and columns. Only valid if ``co_clustering=True``.
    """

    def __init__(self, n_clusters: int = 8, embedding_dimension: int = 16, l2normalization: bool = True,
                 co_clustering: bool = True):
        SpectralClustering.__init__(self, n_clusters, embedding_dimension, l2normalization)
        self.co_clustering = co_clustering
        self.row_labels_ = None
        self.col_labels_ = None
        self.labels_ = None

    def fit(self, biadjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'BiSpectralClustering':
        """Apply embedding method followed by clustering to the graph.

        Parameters
        ----------
        biadjacency:
            Biadjacency matrix of the graph.

        Returns
        -------
        self: :class:`BiSpectralClustering`

        """
        biadjacency = check_format(biadjacency)
        n1, n2 = biadjacency.shape

        bispectral = BiSpectral(self.embedding_dimension).fit(biadjacency)

        if self.co_clustering:
            embedding = bispectral.embedding_
        else:
            embedding = bispectral.row_embedding_

        if self.l2normalization:
            norm = np.linalg.norm(embedding, axis=1)
            norm[norm == 0.] = 1
            embedding /= norm[:, np.newaxis]

        kmeans = KMeans(self.n_clusters)
        kmeans.fit(embedding)

        if self.co_clustering:
            self.row_labels_ = kmeans.labels_[:n1]
            self.col_labels_ = kmeans.labels_[n1:]
            self.labels_ = kmeans.labels_

        else:
            self.row_labels_ = kmeans.labels_
            self.labels_ = kmeans.labels_

        return self
