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

from sknetwork.embedding import BaseEmbedding, Spectral
from sknetwork.clustering.kmeans import get_embedding
from sknetwork.hierarchy.base import BaseHierarchy
from sknetwork.utils.check import check_format
from sknetwork.utils.ward import WardDense


class Ward(BaseHierarchy):
    """Hierarchical clustering by the Ward method.

    Parameters
    ----------
    embedding_method :
        Embedding method (default = Spectral embedding in dimension 10).
    co_cluster :
        If ``True``, co-cluster rows and columns, considered as different nodes (default = ``False``).

    Attributes
    ----------
    dendrogram_ :
        Dendrogram of the graph.
    dendrogram_row_ :
        Dendrogram for the rows, for bipartite graphs.
    dendrogram_col_ :
        Dendrogram for the columns, for bipartite graphs.
    dendrogram_full_ :
        Dendrogram for both rows and columns, indexed in this order, for bipartite graphs.

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
    def __init__(self, embedding_method: BaseEmbedding = Spectral(10), co_cluster: bool = False):
        super(Ward, self).__init__()
        self.embedding_method = embedding_method
        self.co_cluster = co_cluster
        self.bipartite = None

    def fit(self, input_matrix: Union[sparse.csr_matrix, np.ndarray]) -> 'Ward':
        """Applies embedding method followed by the Ward algorithm.

        Parameters
        ----------
        input_matrix :
            Adjacency matrix or biadjacency matrix of the graph.

        Returns
        -------
        self: :class:`Ward`
        """
        self._init_vars()

        # input
        check_format(input_matrix)

        # embedding
        embedding, self.bipartite = get_embedding(input_matrix, self.embedding_method, self.co_cluster)

        # clustering
        ward = WardDense()
        self.dendrogram_ = ward.fit_transform(embedding)

        # output
        if self.co_cluster:
            self._split_vars(input_matrix.shape)

        return self
