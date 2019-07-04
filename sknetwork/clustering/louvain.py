#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 2, 2018
@author: Nathan de Lara <ndelara@enst.fr>
@author: Quentin Lutz <qlutz@enst.fr>
@author: Thomas Bonald <bonald@enst.fr>
"""

from sknetwork.utils.checks import *
from sknetwork.utils.adjacency_formats import *
from sknetwork.utils.algorithm_base_class import Algorithm
from sknetwork import njit


def membership_matrix(labels: np.ndarray) -> sparse.csr_matrix:
    """Builds a n x k matrix whose lines are one-hot vectors representing the cluster assignments of the samples.

    Parameters
    ----------
    labels:
        partition of the samples.

    Returns
    -------
    membership: sparse.csr_matrix

    """
    n_samp = len(labels)
    return sparse.csr_matrix((np.ones(n_samp), (np.arange(n_samp), labels)))


class NormalizedGraph:
    """
    A class of graphs suitable for the Louvain algorithm. Each node represents a cluster.

    Parameters
    ----------
    adjacency :
        Adjacency matrix of the graph.
    weights :
        Distribution of node weights (sums to 1), used in the second term of modularity.

    Attributes
    ----------
    n_nodes: int
        Number of nodes.
    norm_adjacency: sparse.csr_matrix
        Normalized adjacency matrix (sums to 1).
    node_probs : np.ndarray
        Distribution of node weights (sums to 1).
    """

    def __init__(self, adjacency: sparse.csr_matrix, weights: Union[str, np.ndarray] = 'degree',
                 feature_weights: Union[None, 'str', np.ndarray] = 'degree'):
        self.n_nodes, self.n_features = adjacency.shape
        self.norm_adjacency = adjacency / adjacency.data.sum()
        self.node_probs = check_probs(weights, adjacency)
        if feature_weights is not None:
            self.feat_probs = check_probs(feature_weights, adjacency.T)
        else:
            self.feat_probs = None

    def aggregate(self, membership: Union[sparse.csr_matrix, np.ndarray],
                  feat_membership: Union[None, sparse.csr_matrix, np.ndarray] = None):
        """Aggregates nodes belonging to the same cluster.

        Parameters
        ----------
        membership:
            Partition of the nodes (lines of the adjacency).
        feat_membership:
            Partition of the columns.

        Returns
        -------
        The aggregated graph
        """
        if membership.shape[0] != self.n_nodes:
            raise ValueError('The size of the partition must match the number of nodes.')
        elif type(membership) == np.ndarray:
            membership = membership_matrix(membership)

        if feat_membership is not None:
            if feat_membership.shape[0] != self.n_features:
                raise ValueError('The number of feature labels must match the number of columns.')
            if self.feat_probs is None:
                raise ValueError('This graph does not have a feat_probs attribute.')
            elif type(feat_membership) == np.ndarray:
                feat_membership = membership_matrix(feat_membership)

            self.norm_adjacency = membership.T.dot(self.norm_adjacency.dot(feat_membership)).tocsr()
            self.feat_probs = np.array(feat_membership.T.dot(self.feat_probs).T)

        else:
            self.norm_adjacency = membership.T.dot(self.norm_adjacency.dot(membership)).tocsr()
            if self.feat_probs is not None:
                self.feat_probs = np.array(membership.T.dot(self.feat_probs).T)

        self.node_probs = np.array(membership.T.dot(self.node_probs).T)
        self.n_nodes, self.n_features = self.norm_adjacency.shape
        return self


class Optimizer(Algorithm):
    """A generic optimization algorithm.

    Attributes
    ----------
    score_: float
        Total increase of the objective function.
    labels_: np.ndarray
        Cluster index of each node.
    """
    def __init__(self):
        self.score_ = None
        self.labels_ = None

    def fit(self, graph: NormalizedGraph):
        """Fit the clusters to the objective function.

         Parameters
         ----------
         graph:
             Graph to cluster.

         Returns
         -------
         self: :class:̀Optimizer`

         """
        return self


@njit
def fit_core(resolution: float, tol: float, n_nodes: int, ou_node_probs: np.ndarray,
             in_node_probs: np.ndarray, self_loops: np.ndarray, data: np.ndarray, indices: np.ndarray,
             indptr: np.ndarray) -> (np.ndarray, float):
    """

    Parameters
    ----------
    resolution:
        Resolution parameter (positive).
    tol:
        Minimum increase in modularity to enter a new optimization pass.
    n_nodes:
        Number of nodes.
    ou_node_probs:
        Distribution of node weights based on their out-edges (sums to 1).
    in_node_probs:
        Distribution of node weights based on their in-edges (sums to 1).
    self_loops:
        Weights of self loops.
    data:
        CSR format data array of the normalized adjacency matrix.
    indices:
        CSR format index array of the normalized adjacency matrix.
    indptr:
        CSR format index pointer array of the normalized adjacency matrix.

    Returns
    -------
    labels:
        Cluster index of each node
    total_increase:
        Score of the clustering (total increase in modularity)
    """
    increase: bool = True
    total_increase: float = 0

    labels: np.ndarray = np.arange(n_nodes)
    ou_clusters_weights: np.ndarray = ou_node_probs.copy()
    in_clusters_weights: np.ndarray = in_node_probs.copy()

    nodes = np.arange(n_nodes)
    while increase:
        increase = False
        pass_increase: float = 0

        for node in nodes:
            node_cluster = labels[node]
            neighbors: np.ndarray = indices[indptr[node]:indptr[node + 1]]
            weights: np.ndarray = data[indptr[node]:indptr[node + 1]]

            neighbor_clusters_weights = np.zeros(n_nodes)
            for i, neighbor in enumerate(neighbors):
                neighbor_clusters_weights[labels[neighbor]] += weights[i]

            unique_clusters = set(labels[neighbors])
            unique_clusters.discard(node_cluster)

            ou_ratio = resolution * ou_node_probs[node]
            in_ratio = resolution * in_node_probs[node]
            if len(unique_clusters):
                exit_delta: float = 2 * (neighbor_clusters_weights[node_cluster] - self_loops[node])
                exit_delta -= ou_ratio * (in_clusters_weights[node_cluster] - in_node_probs[node])
                exit_delta -= in_ratio * (ou_clusters_weights[node_cluster] - ou_node_probs[node])

                best_delta: float = 0
                best_cluster = node_cluster

                for cluster in unique_clusters:
                    delta: float = 2 * neighbor_clusters_weights[cluster]
                    delta -= ou_ratio * in_clusters_weights[cluster]
                    delta -= in_ratio * ou_clusters_weights[cluster]

                    local_delta = delta - exit_delta
                    if local_delta > best_delta:
                        best_delta = local_delta
                        best_cluster = cluster

                if best_delta > 0:
                    pass_increase += best_delta
                    ou_clusters_weights[node_cluster] -= ou_node_probs[node]
                    in_clusters_weights[node_cluster] -= in_node_probs[node]
                    ou_clusters_weights[best_cluster] += ou_node_probs[node]
                    in_clusters_weights[best_cluster] += in_node_probs[node]
                    labels[node] = best_cluster

        total_increase += pass_increase
        if pass_increase > tol:
            increase = True
    return labels, total_increase


class GreedyModularity(Optimizer):
    """
    A greedy directed modularity optimizer.

    Attributes
    ----------
    resolution:
        modularity resolution
    tol:
        minimum bimodularity increase to enter a new optimization pass
    engine: str
        ``'default'``, ``'python'`` or ``'numba'``. If ``'default'``, it will tests if numba is available.

    """

    def __init__(self, resolution: float = 1, tol: float = 1e-3, engine: str = 'default'):
        Optimizer.__init__(self)
        self.resolution = resolution
        self.tol = tol
        self.engine = check_engine(engine)

    def fit(self, graph: NormalizedGraph):
        """Iterates over the nodes of the graph and moves them to the cluster of highest increase among their neighbors.

        Parameters
        ----------
        graph:
            the graph to cluster

        Returns
        -------
        self: :class:`Optimizer`
        """

        ou_node_probs = graph.node_probs
        if graph.feat_probs is not None:
            in_node_probs = graph.feat_probs
        else:
            in_node_probs = ou_node_probs

        adjacency = 0.5 * directed2undirected(graph.norm_adjacency)

        self_loops = adjacency.diagonal()

        indptr: np.ndarray = adjacency.indptr
        indices: np.ndarray = adjacency.indices
        data: np.ndarray = adjacency.data

        if self.engine == 'python':
            increase: bool = True
            total_increase: float = 0.
            labels: np.ndarray = np.arange(graph.n_nodes)

            ou_clusters_weights: np.ndarray = ou_node_probs.copy()
            in_clusters_weights: np.ndarray = in_node_probs.copy()

            while increase:
                increase = False
                pass_increase: float = 0.

                for node in range(graph.n_nodes):
                    node_cluster: int = labels[node]
                    neighbors: np.ndarray = indices[indptr[node]:indptr[node + 1]]
                    weights: np.ndarray = data[indptr[node]:indptr[node + 1]]

                    neighbors_clusters: np.ndarray = labels[neighbors]
                    unique_clusters: list = list(set(neighbors_clusters.tolist()) - {node_cluster})
                    n_clusters: int = len(unique_clusters)

                    ou_ratio = self.resolution * ou_node_probs[node]
                    in_ratio = self.resolution * in_node_probs[node]
                    if n_clusters > 0:
                        exit_delta: float = 2 * (weights[labels[neighbors] == node_cluster].sum() - self_loops[node])
                        exit_delta -= ou_ratio * (in_clusters_weights[node_cluster] - in_node_probs[node])
                        exit_delta -= in_ratio * (ou_clusters_weights[node_cluster] - ou_node_probs[node])

                        local_delta: np.ndarray = np.full(n_clusters, -exit_delta)

                        for index_cluster, cluster in enumerate(unique_clusters):
                            delta: float = 2 * weights[labels[neighbors] == cluster].sum()
                            delta -= ou_ratio * in_clusters_weights[cluster]
                            delta -= in_ratio * ou_clusters_weights[cluster]

                            local_delta[index_cluster] += delta

                        delta_argmax: int = local_delta.argmax()
                        best_delta: float = local_delta[delta_argmax]
                        if best_delta > 0:
                            pass_increase += best_delta
                            best_cluster = unique_clusters[delta_argmax]

                            ou_clusters_weights[node_cluster] -= ou_node_probs[node]
                            in_clusters_weights[node_cluster] -= in_node_probs[node]
                            ou_clusters_weights[best_cluster] += ou_node_probs[node]
                            in_clusters_weights[best_cluster] += in_node_probs[node]
                            labels[node] = best_cluster

                total_increase += pass_increase
                if pass_increase > self.tol:
                    increase = True

            self.score_ = total_increase
            _, self.labels_ = np.unique(labels, return_inverse=True)

            return self

        elif self.engine == 'numba':
            labels, total_increase = fit_core(self.resolution, self.tol, graph.n_nodes,
                                              ou_node_probs, in_node_probs, self_loops, data, indices, indptr)

            self.score_ = total_increase
            _, self.labels_ = np.unique(labels, return_inverse=True)

            return self

        else:
            raise ValueError('Unknown engine.')


class Louvain(Algorithm):
    """Louvain algorithm for graph clustering in Python (default) and Numba.

    Seek the best partition of the nodes with respect to its modularity by performing local updates in a greedy fashion.
    The modularity of a partition is

    :math:`Q = \\sum_{ij}\\big(\\dfrac{A_{ij}}{w} - \\gamma \\dfrac{d_if_j}{w^2}\\big)\\delta_{ij}`,

    where :math:`\\gamma \\ge 0` is a resolution parameter and :math:`\\delta_{ij} = 1` if nodes :math:`i` and :math:`j`
    belong to the same cluster and :math:`\\delta_{ij} = 0` otherwise.

    Parameters
    ----------
    algorithm:
        The optimization algorithm.
        Requires a fit method.
        Requires `score\\_`  and `labels\\_` attributes.

        If ``'default'``, , it will use a greedy bimodularity optimization algorithm: :class:`GreedyModularity`.
    resolution:
        Resolution parameter.
    tol:
        Minimum increase in the objective function to enter a new optimization pass.
    agg_tol:
        Minimum increase in the objective function to enter a new aggregation pass.
    max_agg_iter:
        Maximum number of aggregations.
        A negative value is interpreted as no limit.
    shuffle_nodes:
        Enables node shuffling before optimization.
    verbose:
        Verbose mode.
    random_state:
        Random number generator or random seed. If None, numpy.random will be used.

    Attributes
    ----------
    labels_: np.ndarray
        Cluster index of each node.
    n_clusters_: int
        The number of clusters in the partition.
    iteration_count_: int
        Total number of aggregations performed.
    aggregate_graph_: sparse.csr_matrix
        Aggregated graph at the end of the algorithm.

    Example
    -------
    >>> louvain = Louvain(GreedyModularity(engine='python'))
    >>> graph = sparse.identity(3, format='csr')
    >>> louvain.fit(graph)
    Louvain(algorithm=GreedyModularity(resolution=1, tol=0.001, engine='python'), agg_tol=0.001, max_agg_iter=-1, \
shuffle_nodes=False, verbose=False)
    >>> louvain.labels_
    array([0, 1, 2])

    References
    ----------
    * Blondel, V. D., Guillaume, J. L., Lambiotte, R., & Lefebvre, E. (2008).
      Fast unfolding of communities in large networks.
      Journal of statistical mechanics: theory and experiment, 2008
      https://arxiv.org/abs/0803.0476

    * Dugué, N., & Perez, A. (2015).
      Directed Louvain: maximizing modularity in directed networks
      (Doctoral dissertation, Université d'Orléans).
      https://hal.archives-ouvertes.fr/hal-01231784/document

    """

    def __init__(self, algorithm: Union[str, Optimizer] = 'default', resolution: float = 1, tol: float = 1e-3,
                 agg_tol: float = 1e-3, max_agg_iter: int = -1, shuffle_nodes: bool = False, verbose: bool = False,
                 random_state: Optional[Union[np.random.RandomState, int]] = None):

        self.random_state = check_random_state(random_state)
        if algorithm == 'default':
            self.algorithm = GreedyModularity(resolution, tol, engine=check_engine('default'))
        elif isinstance(algorithm, Optimizer):
            self.algorithm = algorithm
        else:
            raise TypeError('Algorithm must be \'auto\' or a valid algorithm.')

        if type(max_agg_iter) != int:
            raise TypeError('The maximum number of iterations must be an integer.')
        self.agg_tol = agg_tol
        self.max_agg_iter = max_agg_iter
        self.verbose = verbose
        self.labels_ = None
        self.n_clusters_ = None
        self.iteration_count_ = None
        self.aggregate_graph_ = None
        self.shuffle_nodes = shuffle_nodes

    def fit(self, adjacency: sparse.csr_matrix, weights: Union[str, np.ndarray] = 'degree',
            feature_weights: Union[None, str, np.ndarray] = None) -> 'Louvain':
        """Clustering using chosen Optimizer.

        Parameters
        ----------
        adjacency :
            Adjacency matrix of the graph to cluster.
        weights :
            Probabilities for node sampling in the null model. ``'degree'``, ``'uniform'`` or custom weights.
        feature_weights :
            Probabilities for feature sampling in the null model. ``'degree'``, ``'uniform'`` or custom weights,
            only useful for directed modularity optimization.

        Returns
        -------
        self: :class:`Louvain`
        """
        adjacency = check_format(adjacency)

        if not is_square(adjacency):
            adjacency = bipartite2directed(adjacency)

        nodes = np.arange(adjacency.shape[0])
        if self.shuffle_nodes:
            nodes = self.random_state.permutation(nodes)
            adjacency = adjacency[nodes, :].tocsc()[:, nodes].tocsr()

        graph = NormalizedGraph(adjacency, weights, feature_weights)

        membership = sparse.identity(graph.n_nodes, format='csr')
        increase = True
        iteration_count = 0
        if self.verbose:
            print("Starting with", graph.n_nodes, "nodes.")
        while increase:
            iteration_count += 1

            self.algorithm.fit(graph)

            if self.algorithm.score_ <= self.agg_tol:
                increase = False
            else:
                agg_membership = membership_matrix(self.algorithm.labels_)
                membership = membership.dot(agg_membership)
                graph.aggregate(agg_membership)

                if graph.n_nodes == 1:
                    break
            if self.verbose:
                print("Iteration", iteration_count, "completed with", graph.n_nodes, "clusters and ",
                      self.algorithm.score_, "increment.")
            if iteration_count == self.max_agg_iter:
                break

        self.iteration_count_ = iteration_count
        self.labels_ = membership.indices
        if self.shuffle_nodes:
            reverse = np.empty(nodes.size, nodes.dtype)
            reverse[nodes] = np.arange(nodes.size)
            self.labels_ = self.labels_[reverse]
        self.n_clusters_ = len(set(self.labels_))
        _, self.labels_ = np.unique(self.labels_, return_inverse=True)
        self.aggregate_graph_ = graph.norm_adjacency * adjacency.data.sum()
        return self
