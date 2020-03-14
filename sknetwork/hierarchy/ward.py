#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 2019
@author: Nathan de Lara <ndelara@enst.fr>
@author: Thomas Bonald <bonald@enst.fr>
"""

from typing import Union

import numpy as np
from scipy import sparse

from sknetwork.embedding import BaseEmbedding, GSVD
from sknetwork.hierarchy.base import BaseHierarchy
from sknetwork.utils.ward import WardDense


class Ward(BaseHierarchy):
    """Hierarchical clustering by the Ward method.

    Parameters
    ----------
    embedding_method:
        Embedding method.

    Attributes
    ----------
    dendrogram_:
        Dendrogram.

    """

    def __init__(self, embedding_method: BaseEmbedding = GSVD(10)):
        super(Ward, self).__init__()

        self.embedding_method = embedding_method

    # noinspection DuplicatedCode
    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'Ward':
        """Applies embedding method followed by the Ward algorithm.

        Parameters
        ----------
        adjacency:
            Adjacency matrix of the graph.

        Returns
        -------
        self: :class:`Ward`

        """
        embedding = self.embedding_method.fit_transform(adjacency)
        ward = WardDense()
        ward.fit(embedding)

        self.dendrogram_ = ward.dendrogram_

        return self


class BiWard(Ward):
    """Hierarchical clustering for bipartite graphs by the Ward method.

    Parameters
    ----------
    embedding_dimension:
        Dimension of the embedding on which to apply the hierarchical clustering.
    l2normalization:
        If ``True``, each row of the embedding (and col_embedding) is projected onto the L2-sphere
        before hierarchical clustering.
    cluster_row:
        If ``True``, returns a dendrogram for the rows.
    cluster_col:
        If ``True``, returns a dendrogram for the columns.
    cluster_both:
        If ``True``, also returns a dendrogram for all nodes (co-clustering rows + columns).

    Attributes
    ----------
    dendrogram_row_:
        Dendrogram for the rows.
    dendrogram_col_:
        Dendrogram for the columns.
    dendrogram_:
        Dendrogram (all nodes).

    """

    def __init__(self, embedding_dimension: int = 16, l2normalization: bool = True, cluster_row: bool = True,
                 cluster_col: bool = False, cluster_both: bool = False):
        Ward.__init__(self, embedding_dimension, l2normalization)
        self.cluster_row = cluster_row
        self.cluster_col = cluster_col
        self.cluster_both = cluster_both
        self.dendrogram_row_ = None
        self.dendrogram_col_ = None
        self.dendrogram_ = None

    def fit(self, biadjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'BiWard':
        """Apply embedding method followed by hierarchical clustering to the graph.

        Parameters
        ----------
        biadjacency:
            Biadjacency matrix of the graph.

        Returns
        -------
        self: :class:`BiWard`

        """
        bispectral = BiSpectral(self.embedding_dimension).fit(biadjacency)
        embedding_row = bispectral.embedding_row_
        embedding_col = bispectral.embedding_col_

        if self.l2normalization:
            norm = np.linalg.norm(embedding_row, axis=1)
            norm[norm == 0.] = 1
            embedding_row /= norm[:, np.newaxis]

            norm = np.linalg.norm(embedding_col, axis=1)
            norm[norm == 0.] = 1
            embedding_col /= norm[:, np.newaxis]

        ward = WardDense()
        if self.cluster_row:
            ward.fit(embedding_row)
            self.dendrogram_row_ = ward.dendrogram_

        if self.cluster_col:
            ward.fit(embedding_col)
            self.dendrogram_col_ = ward.dendrogram_

        if self.cluster_both:
            ward.fit(np.vstack((embedding_row, embedding_col)))
            self.dendrogram_ = ward.dendrogram_

        return self
