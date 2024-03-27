"""
Created in March 2024
@author: Laurène David <laurene.david@ip-paris.fr>
"""

from typing import Union

import numpy as np
from scipy import sparse

from sknetwork.clustering import BaseClustering
from sknetwork.ranking import PageRank
from sknetwork.clustering import get_modularity
from sknetwork.classification.pagerank import PageRankClassifier
from sknetwork.utils.format import get_adjacency, directed2undirected


class KCenters(BaseClustering):
    """K-center clustering algorithm. The center of each cluster is obtained by the PageRank algorithm.

    Parameters
    ----------
    n_clusters: int
        Number of clusters.
    directed: bool, default False
        If ``True``, the graph is considered directed.
    center_position: str, default "row"
        Force centers to correspond to the nodes on the rows or columns of the biadjacency matrix.
        Can be ``row``, ``col`` or ``both``. Only considered for bipartite graphs.
    n_init: int, default 5
        Number of reruns of the k-centers algorithm with different centers.
        The run that produce the best modularity is chosen as the final result.
    max_iter: int, default 20
        Maximum number of iterations of the k-centers algorithm for a single run.

    Attributes
    ----------
    labels_: np.ndarray, shape (n_labels,)
        Label of each node.
    labels_row_, labels_col_ : np.ndarray
        Labels of rows and columns, for bipartite graphs.
    centers_: np.ndarray or dict
        Cluster centers. If bipartite graph and center_position is "both", centers is a dict.
    centers_row_, centers_col_: np.ndarray
        Cluster centers of rows and columns, for bipartite graphs.

    Example
    -------
    >>> from sknetwork.clustering import KCenters
    >>> from sknetwork.data import karate_club
    >>> kcenters = KCenters(n_clusters=2)
    >>> adjacency = karate_club()
    >>> labels = kcenters.fit_predict(adjacency)
    >>> len(set(labels))
    2

    """
    def __init__(self, n_clusters: int, directed: bool = False, center_position: str = "row",
                 n_init: int = 5, max_iter: int = 20):
        super(BaseClustering, self).__init__()
        self.n_clusters = n_clusters
        self.directed = directed
        self.center_position = center_position
        self.n_init = n_init
        self.max_iter = max_iter
        self.labels_ = None
        self.centers_ = None
        self.centers_row_ = None
        self.centers_col_ = None
        self.mask_centers = None
        self.bipartite = None

    def _compute_mask_centers(self, input_matrix: Union[sparse.csr_matrix, np.ndarray]):
        """Generate mask to filter nodes that can be cluster centers

        Parameters
        ----------
        input_matrix:
            Adjacency matrix or biadjacency matrix of the graph

        Return
        ----------
        mask: np.array, shape (nb_nodes,)
            Mask for possible cluster centers

        """
        nrow_input = input_matrix.shape[0]
        nrow_adjacency = self.adjacency.shape[0]

        mask = np.zeros(nrow_adjacency, dtype=bool)

        if self.bipartite:
            if self.center_position == "row":
                mask[:nrow_input] = True
            elif self.center_position == "col":
                mask[nrow_input:] = True
            elif self.center_position == "both":
                mask[:] = True
            else:
                raise ValueError('Unknown center position')
        else:
            mask[:] = True

        return mask

    def _init_centers(self, adjacency: Union[sparse.csr_matrix, np.ndarray], n_clusters: int):
        """
        Kcenters ++ initialization to select cluster centers.
        This algorithm is an adaptation of "kmeans ++" for graphs.

        Parameters
        ----------
        adjacency:
            Adjacency matrix of the graph
        n_clusters: int, >0
            Number of centers to initialize

        Returns
        ---------
        centers: np.array, shape (nb_clusters,)
            Initial cluster centers

        """

        nodes = np.arange(adjacency.shape[0])
        centers = []

        # Choose randomly first centers
        center = np.random.choice(nodes, 1)[0]
        centers.append(center)

        for k in range(n_clusters - 1):
            # select nodes that aren't already centers
            mask = np.all(nodes[:, None] != centers, axis=-1)
            mask &= self.mask_centers

            weights = {center: 1}
            ppr_scores = PageRank().fit_predict(adjacency, weights)
            ppr_scores = ppr_scores[mask]

            proba = 1 / (1 + ppr_scores)  # fix division by zero
            proba = proba / np.sum(proba)

            center = np.random.choice(nodes[mask], 1, p=proba)[0]
            centers.append(center)

        centers = np.array(centers)
        return centers

    def fit(self, input_matrix: Union[sparse.csr_matrix, np.ndarray],
            force_bipartite: bool = False) -> "KCenters":
        """Fit algorithm to data

        Parameters
        ---------
        input_matrix :
            Adjacency matrix or biadjacency matrix of the graph.

        force_bipartite :
            If ``True``, force the input matrix to be considered as a biadjacency matrix even if square.

        Returns
        -------
        self : :class:`KCenters`
        """

        if self.n_clusters < 2:
            raise ValueError("The number of clusters must be at least 2.")

        if self.n_init < 1:
            raise ValueError("The n_init parameter must be at least 1.")

        if self.directed:  # directed case
            input_matrix = directed2undirected(input_matrix)

        nrow = input_matrix.shape[0]
        self.adjacency, self.bipartite = get_adjacency(input_matrix, force_bipartite=force_bipartite)
        self.mask_centers = self._compute_mask_centers(input_matrix)

        nodes = np.arange(self.adjacency.shape[0])
        labels_ = []
        centers_ = []
        modularity_ = []

        # Restarts
        for i in range(self.n_init):

            # Initialization
            centers = self._init_centers(self.adjacency, self.n_clusters)
            prev_centers = None
            n_iter = 0

            while np.not_equal(prev_centers, centers).any() and (n_iter < self.max_iter):

                # Assign nodes to centers
                pagerank_c = PageRankClassifier()
                labels_pr = {center: label for label, center in enumerate(centers)}
                labels = pagerank_c.fit_predict(self.adjacency, labels_pr)

                # Find new centers
                ppr = PageRank()
                prev_centers = centers.copy()
                centers__ = []

                for label in np.unique(labels):
                    mask = labels == label
                    mask &= self.mask_centers

                    score = ppr.fit_predict(self.adjacency, weights=mask)
                    score[~mask] = 0
                    centers__.append(nodes[np.argmax(score)])

                centers = centers__.copy()
                n_iter += 1

            # Store results of restarts (labels, modularity and centers)
            if self.bipartite:
                labels_row = labels[:input_matrix.shape[0]]
                labels_col = labels[input_matrix.shape[0]:]
                modularity = get_modularity(input_matrix, labels_row, labels_col)

            else:
                modularity = get_modularity(self.adjacency, labels)

            labels_.append(labels)
            centers_.append(centers)
            modularity_.append(modularity)

        # Select restart with the highest modularity
        idx_max = np.argmax(modularity_)
        self.labels_ = np.array(labels_[idx_max])
        self.centers_ = np.array(centers_[idx_max])

        if self.bipartite:
            self._split_vars(input_matrix.shape)

            # Define centers based on center position
            if self.center_position == "row":
                self.centers_row_ = self.centers_
            elif self.center_position == "col":
                self.centers_col_ = self.centers_ - nrow
                self.centers_ = self.centers_col_
            else:
                self.centers_row_ = self.centers_[self.centers_ < nrow]
                self.centers_col_ = self.centers_[~np.isin(self.centers_, self.centers_row_)] - nrow
                self.centers_ = {"row": self.centers_row_, "col": self.centers_col_}

        return self
