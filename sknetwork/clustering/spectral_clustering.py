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
from sknetwork.utils.checks import check_format, is_symmetric
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

    Attributes
    ----------
    labels_:
        Labels the rows.
    col_labels_:
        Labels of the columns, on valid if ``co_clustering=True``.

    """

    def __init__(self, n_clusters: int = 8, embedding_dimension: int = 16, l2normalization: bool = True):
        self.n_clusters = n_clusters
        self.embedding_dimension = embedding_dimension
        self.l2normalization = l2normalization

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
