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

from sknetwork.embedding import BaseEmbedding, BaseBiEmbedding, GSVD
from sknetwork.hierarchy.base import BaseHierarchy, BaseBiHierarchy
from sknetwork.utils.ward import WardDense


class Ward(BaseHierarchy):
    """Hierarchical clustering by the Ward method.

    * Graphs
    * Digraphs

    Parameters
    ----------
    embedding_method :
        Embedding method (default = GSVD in dimension 10, projected on the unit sphere).

    Examples
    --------
    >>> from sknetwork.hierarchy import Ward
    >>> from sknetwork.data import karate_club
    >>> ward = Ward()
    >>> adjacency = karate_club()
    >>> dendrogram = ward.fit_transform(adjacency)
    >>> dendrogram.shape
    (33, 4)

    References
    ----------
    * Ward, J. H., Jr. (1963). Hierarchical grouping to optimize an objective function.
      Journal of the American Statistical Association.

    * Murtagh, F., & Contreras, P. (2012). Algorithms for hierarchical clustering: an overview.
      Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery.
    """
    def __init__(self, embedding_method: BaseEmbedding = GSVD(10)):
        super(Ward, self).__init__()
        self.embedding_method = embedding_method

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'Ward':
        """Applies embedding method followed by the Ward algorithm.

        Parameters
        ----------
        adjacency :
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


class BiWard(Ward, BaseBiHierarchy):
    """Hierarchical clustering of bipartite graphs by the Ward method.

    * Bigraphs

    Parameters
    ----------
    embedding_method :
        Embedding method (default = GSVD in dimension 10, projected on the unit sphere).
    cluster_col :
        If ``True``, return a dendrogram for the columns (default = ``False``).
    cluster_both :
        If ``True``, return a dendrogram for all nodes (co-clustering rows + columns, default = ``False``).

    Attributes
    ----------
    dendrogram_ :
        Dendrogram for the rows.
    dendrogram_row_ :
        Dendrogram for the rows (copy of **dendrogram_**).
    dendrogram_col_ :
        Dendrogram for the columns.
    dendrogram_full_ :
        Dendrogram for both rows and columns, indexed in this order.

    Examples
    --------
    >>> from sknetwork.hierarchy import BiWard
    >>> from sknetwork.data import movie_actor
    >>> biward = BiWard()
    >>> biadjacency = movie_actor()
    >>> biward.fit_transform(biadjacency).shape
    (14, 4)

    References
    ----------
    * Ward, J. H., Jr. (1963). Hierarchical grouping to optimize an objective function.
      Journal of the American Statistical Association, 58, 236–244.

    * Murtagh, F., & Contreras, P. (2012). Algorithms for hierarchical clustering: an overview.
      Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery, 2(1), 86-97.
    """
    def __init__(self, embedding_method: BaseBiEmbedding = GSVD(10), cluster_col: bool = False,
                 cluster_both: bool = False):
        super(BiWard, self).__init__(embedding_method=embedding_method)
        self.cluster_col = cluster_col
        self.cluster_both = cluster_both

    def fit(self, biadjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'BiWard':
        """Applies the embedding method followed by the Ward algorithm.

        Parameters
        ----------
        biadjacency:
            Biadjacency matrix of the graph.

        Returns
        -------
        self: :class:`BiWard`
        """
        method: BaseBiEmbedding = self.embedding_method
        method.fit(biadjacency)
        embedding_row = method.embedding_row_
        embedding_col = method.embedding_col_

        ward = WardDense()
        ward.fit(embedding_row)
        self.dendrogram_row_ = ward.dendrogram_

        if self.cluster_col:
            ward.fit(embedding_col)
            self.dendrogram_col_ = ward.dendrogram_

        if self.cluster_both:
            ward.fit(np.vstack((embedding_row, embedding_col)))
            self.dendrogram_full_ = ward.dendrogram_

        self.dendrogram_ = self.dendrogram_row_

        return self
