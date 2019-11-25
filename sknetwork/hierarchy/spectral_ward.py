#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

from typing import Union

import numpy as np
from scipy import sparse

from sknetwork.embedding import Spectral, BiSpectral
from sknetwork.hierarchy.base import BaseHierarchy
from sknetwork.utils.ward import Ward


class SpectralWard(BaseHierarchy):
    """Pipeline for spectral Ward hierarchical clustering.

    Parameters
    ----------
    embedding_dimension:
        Dimension of the embedding on which to apply the hierarchical clustering.
    l2normalization:
        If ``True``, each row of the embedding is projected onto the L2-sphere before hierarchical clustering.

    Attributes
    ----------
    dendrogram_:
        Dendrogram.

    """

    def __init__(self, embedding_dimension: int = 16, l2normalization: bool = True):
        super(SpectralWard, self).__init__()

        self.embedding_dimension = embedding_dimension
        self.l2normalization = l2normalization

    # noinspection DuplicatedCode
    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'SpectralWard':
        """Apply embedding method followed by hierarchical clustering to the graph.

        Parameters
        ----------
        adjacency:
            Adjacency matrix of the graph.

        Returns
        -------
        self: :class:`SpectralWard`

        """
        spectral = Spectral(self.embedding_dimension).fit(adjacency)
        embedding = spectral.embedding_

        if self.l2normalization:
            norm = np.linalg.norm(embedding, axis=1)
            norm[norm == 0.] = 1
            embedding /= norm[:, np.newaxis]

        ward = Ward()
        ward.fit(embedding)

        self.dendrogram_ = ward.dendrogram_

        return self


class BiSpectralWard(SpectralWard):
    """Pipeline for spectral Ward hierarchical clustering for bipartite graphs.

    Parameters
    ----------
    embedding_dimension:
        Dimension of the embedding on which to apply the hierarchical clustering.
    l2normalization:
        If ``True``, each row of the embedding (and col_embedding) is projected onto the L2-sphere
        before hierarchical clustering.
    col_clustering:
        If ``True``, returns a dendrogram for the columns.
    co_clustering:
        If ``True``, also returns a dendrogram for all nodes (co-clustering rows + columns).

    Attributes
    ----------
    row_dendrogram_:
        Dendrogram for the rows.
    col_dendrogram_:
        Dendrogram for the columns.
    dendrogram_:
        Dendrogram (all nodes).

    """

    def __init__(self, embedding_dimension: int = 16, l2normalization: bool = True, col_clustering: bool = True,
                 co_clustering: bool = False):
        SpectralWard.__init__(self, embedding_dimension, l2normalization)
        self.col_clustering = col_clustering
        self.co_clustering = co_clustering
        self.row_dendrogram_ = None
        self.col_dendrogram_ = None
        self.dendrogram_ = None

    def fit(self, biadjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'BiSpectralWard':
        """Apply embedding method followed by hierarchical clustering to the graph.

        Parameters
        ----------
        biadjacency:
            Biadjacency matrix of the graph.

        Returns
        -------
        self: :class:`BiSpectralWard`

        """
        bispectral = BiSpectral(self.embedding_dimension).fit(biadjacency)
        row_embedding = bispectral.row_embedding_
        col_embedding = bispectral.col_embedding_

        if self.l2normalization:
            norm = np.linalg.norm(row_embedding, axis=1)
            norm[norm == 0.] = 1
            row_embedding /= norm[:, np.newaxis]

            norm = np.linalg.norm(col_embedding, axis=1)
            norm[norm == 0.] = 1
            col_embedding /= norm[:, np.newaxis]

        ward = Ward()
        ward.fit(row_embedding)
        self.row_dendrogram_ = ward.dendrogram_.copy()

        if self.col_clustering:
            ward.fit(col_embedding)
            self.col_dendrogram_ = ward.dendrogram_.copy()

        if self.co_clustering:
            ward.fit(np.vstack((row_embedding, col_embedding)))
            self.dendrogram_ = ward.dendrogram_

        return self
