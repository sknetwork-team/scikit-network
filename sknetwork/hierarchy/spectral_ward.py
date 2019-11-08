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
from sknetwork.utils.algorithm_base_class import Algorithm
from sknetwork.utils.ward import Ward


class SpectralWard(Algorithm):
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
        self.embedding_dimension = embedding_dimension
        self.l2normalization = l2normalization

        self.dendrogram_ = None

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
    col_hierarchy:
        If ``True``, compute a dendrogram for the embedding of the column besides the one of the rows.

    Attributes
    ----------
    dendrogram_:
        Dendrogram for the rows.
    col_dendrogram_:
        Dendrogram for the columns.

    """
    def __init__(self, embedding_dimension: int = 16, l2normalization: bool = True, col_hierarchy: bool = True):
        SpectralWard.__init__(self, embedding_dimension, l2normalization)
        self.col_hierarchy = col_hierarchy

        self.col_dendrogram_ = None

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'BiSpectralWard':
        """Apply embedding method followed by hierarchical clustering to the graph.

        Parameters
        ----------
        adjacency:
            Adjacency matrix of the graph.

        Returns
        -------
        self: :class:`BiSpectralWard`

        """
        bispectral = BiSpectral(self.embedding_dimension).fit(adjacency)
        embedding = bispectral.embedding_
        col_embedding = bispectral.col_embedding_

        if self.l2normalization:
            norm = np.linalg.norm(embedding, axis=1)
            norm[norm == 0.] = 1
            embedding /= norm[:, np.newaxis]

            norm = np.linalg.norm(col_embedding, axis=1)
            norm[norm == 0.] = 1
            col_embedding /= norm[:, np.newaxis]

        ward = Ward()
        ward.fit(embedding)
        self.dendrogram_ = ward.dendrogram_.copy()

        if self.col_hierarchy:
            ward.fit(col_embedding)
            self.col_dendrogram_ = ward.dendrogram_

        return self
