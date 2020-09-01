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

from sknetwork.clustering.base import BaseClustering, BaseBiClustering
from sknetwork.clustering.postprocess import reindex_labels
from sknetwork.embedding import BaseEmbedding, BaseBiEmbedding, GSVD
from sknetwork.linalg import normalize
from sknetwork.utils.check import check_n_clusters
from sknetwork.utils.kmeans import KMeansDense
from sknetwork.utils.membership import membership_matrix


class KMeans(BaseClustering):
    """K-means applied in the embedding space.

    * Graphs
    * Digraphs

    Parameters
    ----------
    n_clusters :
        Number of desired clusters.
    embedding_method :
        Embedding method (default = GSVD in dimension 10, projected on the unit sphere).
    sort_clusters :
            If ``True``, sort labels in decreasing order of cluster size.
    return_membership :
            If ``True``, return the membership matrix of nodes to each cluster (soft clustering).
    return_aggregate :
            If ``True``, return the adjacency matrix of the graph between clusters.
    Attributes
    ----------
    labels_ : np.ndarray
        Label of each node.
    membership_ : sparse.csr_matrix
        Membership matrix.
    adjacency_ : sparse.csr_matrix
        Adjacency matrix between clusters.

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
    def __init__(self, n_clusters: int = 8, embedding_method: BaseEmbedding = GSVD(10), sort_clusters: bool = True,
                 return_membership: bool = True, return_aggregate: bool = True):
        super(KMeans, self).__init__(sort_clusters=sort_clusters, return_membership=return_membership,
                                     return_aggregate=return_aggregate)
        self.n_clusters = n_clusters
        self.embedding_method = embedding_method

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'KMeans':
        """Apply embedding method followed by K-means.

        Parameters
        ----------
        adjacency:
            Adjacency matrix of the graph.

        Returns
        -------
        self: :class:`KMeans`
        """
        n = adjacency.shape[0]
        check_n_clusters(self.n_clusters, n)

        embedding = self.embedding_method.fit_transform(adjacency)
        kmeans = KMeansDense(self.n_clusters)
        kmeans.fit(embedding)

        if self.sort_clusters:
            labels = reindex_labels(kmeans.labels_)
        else:
            labels = kmeans.labels_

        self.labels_ = labels
        self._secondary_outputs(adjacency)

        return self


class BiKMeans(KMeans, BaseBiClustering):
    """KMeans clustering of bipartite graphs applied in the embedding space.

    * Bigraphs

    Parameters
    ----------
    n_clusters :
        Number of clusters.
    embedding_method :
        Embedding method (default = GSVD in dimension 10, projected on the unit sphere).
    co_cluster :
        If ``True``, co-cluster rows and columns (default = ``False``).
    sort_clusters :
            If ``True``, sort labels in decreasing order of cluster size.
    return_membership :
            If ``True``, return the membership matrix of nodes to each cluster (soft clustering).
    return_aggregate :
            If ``True``, return the biadjacency matrix of the graph between clusters.
    Attributes
    ----------
    labels_ : np.ndarray
        Labels of the rows.
    labels_row_ : np.ndarray
        Labels of the rows (copy of **labels_**).
    labels_col_ : np.ndarray
        Labels of the columns. Only valid if **co_cluster** = `True`.
    membership_ : sparse.csr_matrix
        Membership matrix of the rows.
    membership_row_ : sparse.csr_matrix
        Membership matrix of the rows (copy of **membership_**).
    membership_col_ : sparse.csr_matrix
        Membership matrix of the columns. Only valid if **co_cluster** = `True`.
    biadjacency_ : sparse.csr_matrix
        Biadjacency matrix of the graph between clusters.

    Example
    -------
    >>> from sknetwork.clustering import BiKMeans
    >>> from sknetwork.data import movie_actor
    >>> bikmeans = BiKMeans()
    >>> biadjacency = movie_actor()
    >>> labels = bikmeans.fit_transform(biadjacency)
    >>> len(labels)
    15
    """
    def __init__(self, n_clusters: int = 2, embedding_method: BaseBiEmbedding = GSVD(10), co_cluster: bool = False,
                 sort_clusters: bool = True, return_membership: bool = True, return_aggregate: bool = True):
        super(BiKMeans, self).__init__(sort_clusters=sort_clusters, return_membership=return_membership,
                                       return_aggregate=return_aggregate, n_clusters=n_clusters,
                                       embedding_method=embedding_method)
        self.co_cluster = co_cluster

    def fit(self, biadjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'BiKMeans':
        """Apply embedding method followed by clustering to the graph.

        Parameters
        ----------
        biadjacency:
            Biadjacency matrix of the graph.

        Returns
        -------
        self: :class:`BiKMeans`
        """
        n_row, n_col = biadjacency.shape
        check_n_clusters(self.n_clusters, n_row)

        method = self.embedding_method
        method.fit(biadjacency)

        if self.co_cluster:
            embedding = np.vstack((method.embedding_row_, method.embedding_col_))
        else:
            embedding = method.embedding_

        kmeans = KMeansDense(self.n_clusters)
        kmeans.fit(embedding)

        if self.sort_clusters:
            labels = reindex_labels(kmeans.labels_)
        else:
            labels = kmeans.labels_

        self.labels_ = labels
        if self.co_cluster:
            self._split_vars(n_row)
        else:
            self.labels_row_ = labels

        if self.return_membership:
            membership_row = membership_matrix(self.labels_row_, n_labels=self.n_clusters)
            if self.labels_col_ is not None:
                membership_col = membership_matrix(self.labels_col_, n_labels=self.n_clusters)
                self.membership_row_ = normalize(biadjacency.dot(membership_col))
                self.membership_col_ = normalize(biadjacency.T.dot(membership_row))
            else:
                self.membership_row_ = normalize(biadjacency.dot(biadjacency.T.dot(membership_row)))
            self.membership_ = self.membership_row_

        if self.return_aggregate:
            membership_row = membership_matrix(self.labels_row_, n_labels=self.n_clusters)
            biadjacency_ = sparse.csr_matrix(membership_row.T.dot(biadjacency))
            if self.labels_col_ is not None:
                membership_col = membership_matrix(self.labels_col_, n_labels=self.n_clusters)
                biadjacency_ = biadjacency_.dot(membership_col)
            self.biadjacency_ = biadjacency_

        return self
