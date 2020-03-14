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
    n_components:
        Dimension of the embedding on which to apply the clustering.
    normalize:
        If ``True``, each row of the embedding is projected onto the unit sphere before clustering.

    Attributes
    ----------
    labels_:
        Labels of the rows.

    """

    def __init__(self, n_clusters: int = 8, n_components: int = 16, normalize: bool = True):
        super(SpectralClustering, self).__init__()

        self.n_clusters = n_clusters
        self.n_components = n_components
        self.normalize = normalize

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

        spectral = Spectral(self.n_components).fit(adjacency)
        embedding = spectral.embedding_

        if self.normalize:
            norm = np.linalg.norm(embedding, axis=1)
            norm[norm == 0.] = 1
            embedding /= norm[:, np.newaxis]

        kmeans = KMeans(self.n_clusters)
        kmeans.fit(embedding)

        self.labels_ = kmeans.labels_

        return self


class BiSpectralClustering(SpectralClustering):
    """KMeans clustering.

    Parameters
    ----------
    n_clusters:
        Number of clusters.
    n_components:
        Dimension of the embedding.
    normalize:
        If ``True``, embed the nodes on the unit sphere.
    co_cluster:
        If ``True``, jointly clusters rows and columns of the biadjacency matrix.
        Otherwise, only cluster the rows.

    Attributes
    ----------
    labels_: np.ndarray
        Labels of the rows.
    labels_row_: np.ndarray
        Labels of the rows (copy of labels_).
    labels_col_: np.ndarray
        Labels of the columns. Only valid if ``co_cluster=True``.
    """

    def __init__(self, n_clusters: int = 8, n_components: int = 16, normalize: bool = True,
                 co_cluster: bool = True):
        SpectralClustering.__init__(self, n_clusters, n_components, normalize)
        self.co_cluster = co_cluster
        self.labels_ = None
        self.labels_row_ = None
        self.labels_col_ = None

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

        bispectral = BiSpectral(self.n_components, normalize=self.normalize).fit(biadjacency)

        if self.co_cluster:
            embedding = np.vstack((bispectral.embedding_row_, bispectral.embedding_col_))
        else:
            embedding = bispectral.embedding_row_

        kmeans = KMeans(self.n_clusters)
        kmeans.fit(embedding)

        if self.co_cluster:
            self.labels_ = kmeans.labels_[:n1]
            self.labels_row_ = kmeans.labels_[:n1]
            self.labels_col_ = kmeans.labels_[n1:]

        else:
            self.labels_ = kmeans.labels_
            self.labels_row_ = kmeans.labels_

        return self
