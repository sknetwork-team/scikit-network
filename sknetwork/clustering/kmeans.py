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
from sknetwork.clustering.post_processing import reindex_clusters
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
    sort_cluster :
            If ``True``, sort labels in decreasing order of cluster size.
    """

    def __init__(self, n_clusters: int = 8, embedding_method: BaseEmbedding = GSVD(10), sort_cluster: bool = True):
        super(KMeans, self).__init__()

        self.n_clusters = n_clusters
        self.embedding_method = embedding_method
        self.sort_cluster = sort_cluster

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
        embedding = self.embedding_method.fit_transform(adjacency)
        kmeans = KMeansDense(self.n_clusters)
        kmeans.fit(embedding)

        if self.sort_cluster:
            self.labels_ = reindex_clusters(kmeans.labels_)
        else:
            self.labels_ = kmeans.labels_

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

    Attributes
    ----------
    labels_ : np.ndarray
        Labels of the rows.
    labels_row_ : np.ndarray
        Labels of the rows (copy of **labels_**).
    labels_col_ : np.ndarray
        Labels of the columns. Only valid if **cluster_both** = `True`.
    """

    def __init__(self, n_clusters: int = 8, embedding_method: BaseEmbedding = GSVD(10), cluster_both: bool = False):
        KMeans.__init__(self, n_clusters, embedding_method)

        if not hasattr(embedding_method, 'embedding_col_'):
            raise ValueError('The embedding method is not valid for bipartite graphs.')

        self.cluster_both = cluster_both

        self.labels_ = None
        self.labels_row_ = None
        self.labels_col_ = None

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

        method = self.embedding_method
        method.fit(biadjacency)

        if self.cluster_both:
            embedding = np.vstack((method.embedding_row_, method.embedding_col_))
        else:
            embedding = method.embedding_row_

        kmeans = KMeansDense(self.n_clusters)
        kmeans.fit(embedding)

        if self.sort_cluster:
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

        return self
