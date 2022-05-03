#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 2019
@author: Nathan de Lara <ndelara@enst.fr>
@author: Thomas Bonald <bonald@enst.fr>
"""
from typing import Union, Tuple

import numpy as np
from scipy import sparse

from sknetwork.clustering.base import BaseClustering
from sknetwork.clustering.postprocess import reindex_labels
from sknetwork.embedding.base import BaseEmbedding
from sknetwork.embedding.spectral import Spectral
from sknetwork.utils.format import is_square
from sknetwork.utils.check import check_n_clusters, check_format
from sknetwork.utils.kmeans import KMeansDense


def get_embedding(input_matrix: Union[sparse.csr_matrix, np.ndarray], method: BaseEmbedding,
                  co_embedding: bool = False) -> Tuple[np.ndarray, bool]:
    """Return the embedding of the input_matrix.
    Parameters
    ----------
    input_matrix :
        Adjacency matrix of biadjacency matrix of the graph.
    method :
        Embedding method.
    co_embedding : bool
        If ``True``, co-embedding of rows and columns.
        Otherwise, do it only if the input matrix is not square or not symmetric with ``allow_directed=False``.
    """
    bipartite = (not is_square(input_matrix)) or co_embedding
    if co_embedding:
        try:
            method.fit(input_matrix, force_bipartite=True)
        except:
            method.fit(input_matrix)
        embedding = np.vstack((method.embedding_row_, method.embedding_col_))
    else:
        method.fit(input_matrix)
        embedding = method.embedding_
    return embedding, bipartite


class KMeans(BaseClustering):
    """K-means clustering applied in the embedding space.

    Parameters
    ----------
    n_clusters :
        Number of desired clusters (default = 2).
    embedding_method :
        Embedding method (default = Spectral embedding in dimension 10).
    co_cluster :
        If ``True``, co-cluster rows and columns, considered as different nodes (default = ``False``).
    sort_clusters :
            If ``True``, sort labels in decreasing order of cluster size.
    return_membership :
            If ``True``, return the membership matrix of nodes to each cluster (soft clustering).
    return_aggregate :
            If ``True``, return the adjacency matrix of the graph between clusters.
    Attributes
    ----------
    labels_ : np.ndarray
        Labels of the nodes.
    labels_row_ : np.ndarray
        Labels of the rows (for bipartite graphs).
    labels_col_ : np.ndarray
        Labels of the columns (for bipartite graphs).
    membership_ : sparse.csr_matrix
        Membership matrix of the nodes, shape (n_nodes, n_clusters).
    membership_row_ : sparse.csr_matrix
        Membership matrix of the rows (for bipartite graphs).
    membership_col_ : sparse.csr_matrix
        Membership matrix of the columns (for bipartite graphs).
    aggregate_ : sparse.csr_matrix
        Aggregate adjacency matrix or biadjacency matrix between clusters.

    Example
    -------
    >>> from sknetwork.clustering import KMeans
    >>> from sknetwork.data import karate_club
    >>> kmeans = KMeans(n_clusters=3)
    >>> adjacency = karate_club()
    >>> labels = kmeans.fit_transform(adjacency)
    >>> len(set(labels))
    3
    """
    def __init__(self, n_clusters: int = 2, embedding_method: BaseEmbedding = Spectral(10), co_cluster: bool = False,
                 sort_clusters: bool = True, return_membership: bool = True, return_aggregate: bool = True):
        super(KMeans, self).__init__(sort_clusters=sort_clusters, return_membership=return_membership,
                                     return_aggregate=return_aggregate)
        self.n_clusters = n_clusters
        self.embedding_method = embedding_method
        self.co_cluster = co_cluster
        self.bipartite = None

    def fit(self, input_matrix: Union[sparse.csr_matrix, np.ndarray]) -> 'KMeans':
        """Apply embedding method followed by K-means.

        Parameters
        ----------
        input_matrix :
            Adjacency matrix or biadjacency matrix of the graph.

        Returns
        -------
        self: :class:`KMeans`
        """
        self._init_vars()

        # input
        check_format(input_matrix)
        if self.co_cluster:
            check_n_clusters(self.n_clusters, np.sum(input_matrix.shape))
        else:
            check_n_clusters(self.n_clusters, input_matrix.shape[0])

        # embedding
        embedding, self.bipartite = get_embedding(input_matrix, self.embedding_method, self.co_cluster)

        # clustering
        kmeans = KMeansDense(self.n_clusters)
        kmeans.fit(embedding)

        # sort
        if self.sort_clusters:
            labels = reindex_labels(kmeans.labels_)
        else:
            labels = kmeans.labels_

        # output
        self.labels_ = labels
        if self.co_cluster:
            self._split_vars(input_matrix.shape)
        self._secondary_outputs(input_matrix)

        return self
