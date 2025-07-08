"""
Created in March 2024
@author: Laurène David <laurene.david@ip-paris.fr>
@author: Thomas Bonald <bonald@enst.fr>
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
    n_clusters : int
        Number of clusters.
    directed : bool, default False
        If ``True``, the graph is considered directed.
    center_position : str, default "row"
        Force centers to correspond to the nodes on the rows or columns of the biadjacency matrix.
        Can be ``row``, ``col`` or ``both``. Only considered for bipartite graphs.
    n_init : int, default 5
        Number of reruns of the k-centers algorithm with different centers.
        The run that produce the best modularity is chosen as the final result.
    max_iter : int, default 20
        Maximum number of iterations of the k-centers algorithm for a single run.

    Attributes
    ----------
    labels\_ : np.ndarray, shape (n_nodes,)
        Label of each node.
    probs\_ : sparse.csr_matrix, shape (n_nodes, n_labels)
        Probability distribution over labels.
    aggregate\_ : sparse.csr_matrix
        Aggregate adjacency matrix or biadjacency matrix between clusters.

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
    def __init__(self, n_clusters: int, directed: bool = False, center_position: str = "row", n_init: int = 5,
                 max_iter: int = 20):
        super(BaseClustering, self).__init__()
        self.n_clusters = n_clusters
        self.directed = directed
        self.bipartite = None
        self.center_position = center_position
        self.n_init = n_init
        self.max_iter = max_iter
        self.labels_ = None
        self.centers_ = None
        self.centers_row_ = None
        self.centers_col_ = None

    def _compute_mask_centers(self, input_matrix: Union[sparse.csr_matrix, np.ndarray]):
        """Generate mask to filter nodes that can be cluster centers.

        Parameters
        ----------
        input_matrix :
            Adjacency matrix or biadjacency matrix of the graph.

        Return
        ------
        mask : np.array, shape (n_nodes,)
            Mask for possible cluster centers.

        """
        n_row, n_col = input_matrix.shape
        if self.bipartite:
            n_nodes = n_row + n_col
            mask = np.zeros(n_nodes, dtype=bool)
            if self.center_position == "row":
                mask[:n_row] = True
            elif self.center_position == "col":
                mask[n_row:] = True
            elif self.center_position == "both":
                mask[:] = True
            else:
                raise ValueError('Unknown center position')
        else:
            mask = np.ones(n_row, dtype=bool)

        return mask

    @staticmethod
    def _init_centers(adjacency: Union[sparse.csr_matrix, np.ndarray], mask: np.ndarray, n_clusters: int):
        """
        Kcenters++ initialization to select cluster centers.
        This algorithm is an adaptation of the Kmeans++ algorithm to graphs.

        Parameters
        ----------
        adjacency :
            Adjacency matrix of the graph.
        mask :
            Initial mask for allowed positions of centers.
        n_clusters : int
            Number of centers to initialize.

        Returns
        ---------
        centers : np.array, shape (n_clusters,)
            Initial cluster centers.
        """
        mask = mask.copy()
        n_nodes = adjacency.shape[0]
        nodes = np.arange(n_nodes)
        centers = []

        # Choose the first center uniformly at random
        center = np.random.choice(nodes[mask])
        mask[center] = 0
        centers.append(center)

        pagerank = PageRank()
        weights = {center: 1}

        for k in range(n_clusters - 1):
            # select nodes that are far from existing centers
            ppr_scores = pagerank.fit_predict(adjacency, weights)
            ppr_scores = ppr_scores[mask]

            if min(ppr_scores) == 0:
                center = np.random.choice(nodes[mask][ppr_scores == 0])
            else:
                probs = 1 / ppr_scores
                probs = probs / np.sum(probs)
                center = np.random.choice(nodes[mask], p=probs)

            mask[center] = 0
            centers.append(center)
            weights.update({center: 1})

        centers = np.array(centers)
        return centers

    def fit(self, input_matrix: Union[sparse.csr_matrix, np.ndarray], force_bipartite: bool = False) -> "KCenters":
        """Compute the clustering of the graph by k-centers.

        Parameters
        ----------
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

        if self.directed:
            input_matrix = directed2undirected(input_matrix)

        adjacency, self.bipartite = get_adjacency(input_matrix, force_bipartite=force_bipartite)
        n_row = input_matrix.shape[0]
        n_nodes = adjacency.shape[0]
        nodes = np.arange(n_nodes)

        mask = self._compute_mask_centers(input_matrix)
        if self.n_clusters > np.sum(mask):
            raise ValueError("The number of clusters is to high. This might be due to the center_position parameter.")

        pagerank_clf = PageRankClassifier()
        pagerank = PageRank()

        labels_ = []
        centers_ = []
        modularity_ = []

        # Restarts
        for i in range(self.n_init):

            # Initialization
            centers = self._init_centers(adjacency, mask, self.n_clusters)
            prev_centers = None
            labels = None
            n_iter = 0

            while not np.equal(prev_centers, centers).all() and (n_iter < self.max_iter):

                # Assign nodes to centers
                labels_center = {center: label for label, center in enumerate(centers)}
                labels = pagerank_clf.fit_predict(adjacency, labels_center)

                # Find new centers
                prev_centers = centers.copy()
                new_centers = []

                for label in np.unique(labels):
                    mask_cluster = labels == label
                    mask_cluster &= mask
                    scores = pagerank.fit_predict(adjacency, weights=mask_cluster)
                    scores[~mask_cluster] = 0
                    new_centers.append(nodes[np.argmax(scores)])

                n_iter += 1

            # Store results
            if self.bipartite:
                labels_row = labels[:n_row]
                labels_col = labels[n_row:]
                modularity = get_modularity(input_matrix, labels_row, labels_col)
            else:
                modularity = get_modularity(adjacency, labels)

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
                self.centers_col_ = self.centers_ - n_row
            else:
                self.centers_row_ = self.centers_[self.centers_ < n_row]
                self.centers_col_ = self.centers_[~np.isin(self.centers_, self.centers_row_)] - n_row

        return self
