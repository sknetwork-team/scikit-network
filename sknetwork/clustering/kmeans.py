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
from sknetwork.clustering.postprocess import membership_matrix, reindex_clusters
from sknetwork.embedding import BaseEmbedding, GSVD
from sknetwork.utils.kmeans import KMeansDense


class KMeans(BaseClustering):
    """K-means applied in the embedding space.

    Parameters
    ----------
    n_clusters:
        Number of desired clusters.
    embedding_method:
        Embedding method (default = GSVD in dimension 10, projected on the unit sphere).
    sort_clusters :
            If ``True``, sort labels in decreasing order of cluster size.
    return_graph :
            If ``True``, return the adjacency matrix of the graph between clusters.
    Attributes
    ----------
    labels_ : np.ndarray
        Label of each node.
    adjacency_ : sparse.csr_matrix
        Adjacency matrix between clusters. Only valid if **return_graph** = `True`.

    Example
    -------
    >>> from sknetwork.data import karate_club
    >>> adjacency = karate_club()
    >>> kmeans = KMeans(n_clusters=3)
    >>> len(set(kmeans.fit_transform(adjacency)))
    3

    """

    def __init__(self, n_clusters: int = 8, embedding_method: BaseEmbedding = GSVD(10), sort_clusters: bool = True,
                 return_graph: bool = False):
        super(KMeans, self).__init__()

        self.n_clusters = n_clusters
        self.embedding_method = embedding_method
        self.sort_clusters = sort_clusters
        self.return_graph = return_graph

        self.adjacency_ = None

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
        if self.n_clusters > adjacency.shape[0]:
            raise ValueError('The number of clusters exceeds the number of nodes.')

        embedding = self.embedding_method.fit_transform(adjacency)
        kmeans = KMeansDense(self.n_clusters)
        kmeans.fit(embedding)

        if self.sort_clusters:
            labels = reindex_clusters(kmeans.labels_)
        else:
            labels = kmeans.labels_
        self.labels_ = labels

        if self.return_graph:
            membership = membership_matrix(labels)
            self.adjacency_ = membership.T.dot(adjacency.dot(membership))

        return self


class BiKMeans(KMeans):
    """KMeans co-clustering applied in the embedding space.

    Parameters
    ----------
    n_clusters :
        Number of clusters.
    embedding_method :
        Embedding method (default = GSVD in dimension 10, projected on the unit sphere).
    cluster_both :
        If ``True``, co-cluster rows and columns (default = ``False``).
    sort_clusters :
            If ``True``, sort labels in decreasing order of cluster size.
    return_graph :
            If ``True``, return the biadjacency matrix of the graph between clusters.
    Attributes
    ----------
    labels_ : np.ndarray
        Labels of the rows.
    labels_row_ : np.ndarray
        Labels of the rows (copy of **labels_**).
    labels_col_ : np.ndarray
        Labels of the columns. Only valid if **cluster_both** = `True`.
    biadjacency_ : sparse.csr_matrix
        Biadjacency matrix of the graph between clusters. Only valid if **return_graph** = `True`.

    Example
    -------
    >>> from sknetwork.data import movie_actor
    >>> biadjacency = movie_actor()
    >>> bikmeans = BiKMeans(n_clusters=3)
    >>> len(set(bikmeans.fit_transform(biadjacency)))
    3
    """

    def __init__(self, n_clusters: int = 2, embedding_method: BaseEmbedding = GSVD(10), cluster_both: bool = False,
                 sort_clusters: bool = True, return_graph: bool = False):
        KMeans.__init__(self, n_clusters, embedding_method, sort_clusters, return_graph)

        if not hasattr(embedding_method, 'embedding_col_'):
            raise ValueError('The embedding method is not valid for bipartite graphs.')

        self.cluster_both = cluster_both

        self.labels_ = None
        self.labels_row_ = None
        self.labels_col_ = None
        self.biadjacency_ = None

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
        n1, n2 = biadjacency.shape

        if self.n_clusters > n1:
            raise ValueError('The number of clusters exceeds the number of rows.')

        method = self.embedding_method
        method.fit(biadjacency)

        if self.cluster_both:
            embedding = np.vstack((method.embedding_row_, method.embedding_col_))
        else:
            embedding = method.embedding_row_

        kmeans = KMeansDense(self.n_clusters)
        kmeans.fit(embedding)

        if self.sort_clusters:
            labels = reindex_clusters(kmeans.labels_)
        else:
            labels = kmeans.labels_

        if self.cluster_both:
            self.labels_ = labels[:n1]
            self.labels_row_ = labels[:n1]
            self.labels_col_ = labels[n1:]
        else:
            self.labels_ = labels
            self.labels_row_ = labels

        if self.return_graph:
            membership_row = membership_matrix(self.labels_row_)
            biadjacency_ = sparse.csr_matrix(membership_row.T.dot(biadjacency))
            if self.labels_col_ is not None:
                membership_col = membership_matrix(self.labels_col_)
                biadjacency_ = biadjacency_.dot(membership_col)
            self.biadjacency_ = biadjacency_

        return self
