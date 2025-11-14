# distutils: language = c++
# cython: language_level=3
"""
Created on March 2019
@author: Thomas Bonald <bonald@enst.fr>
@author: Bertrand Charpentier <bertrand.charpentier@live.fr>
@author: Quentin Lutz <qlutz@enst.fr>
"""
import numpy as np
cimport numpy as np

cimport cython

from libcpp.vector cimport vector

from typing import Union

from scipy import sparse

from sknetwork.hierarchy.base import BaseHierarchy
from sknetwork.hierarchy.postprocess import reorder_dendrogram
from sknetwork.utils.format import check_format, get_adjacency, directed2undirected
from sknetwork.utils.check import get_probs, is_symmetric


cdef class AggregateGraph:
    """A class of graphs suitable for aggregation. Each node represents a cluster.

    Parameters
    ----------
    out_weights :
        Out-weights (sums to 1).
    in_weights :
        In-weights (sums to 1).
    data :
        CSR format data array of the normalized adjacency matrix.
    indices :
        CSR format index array of the normalized adjacency matrix.
    indptr :
        CSR format index pointer array of the normalized adjacency matrix.

    Attributes
    ----------
    neighbors : dict[dict]
        Dictionary of dictionary of edge weights.
    next_cluster : int
        Index of the next cluster (resulting from aggregation).
    cluster_sizes : dict
        Dictionary of cluster sizes.
    cluster_out_weights : dict
        Dictionary of cluster out-weights (sums to 1).
    cluster_in_weights : dict
        Dictionary of cluster in-weights (sums to 1).
    """
    cdef public int next_cluster
    cdef public dict neighbors
    cdef public dict tmp
    cdef dict cluster_sizes
    cdef public dict cluster_out_weights
    cdef public dict cluster_in_weights

    def __init__(self, double[:] out_weights, double[:] in_weights, double[:] data, int[:] indices,
                 int[:] indptr):
        cdef int n = indptr.shape[0] - 1
        cdef float total_weight = np.sum(data)
        cdef int i
        cdef int j

        self.next_cluster = n
        self.neighbors = {}
        for i in range(n):
            # normalize so that the sum of edge weights is equal to 1
            self.neighbors[i] = {}
            for j in range(indptr[i], indptr[i + 1]):
                self.neighbors[i][indices[j]] = data[j] / total_weight

        cluster_sizes = {}
        cluster_out_weights = {}
        cluster_in_weights = {}
        for i in range(n):
            cluster_sizes[i] = 1
            cluster_out_weights[i] = out_weights[i]
            cluster_in_weights[i] = in_weights[i]
        self.cluster_sizes = cluster_sizes
        self.cluster_out_weights = cluster_out_weights
        self.cluster_in_weights = cluster_in_weights

    cdef float similarity(self, int node1, int node2):
        """Similarity of two nodes.

        Parameters
        ----------
        node1, node2 :
            Nodes.

        Returns
        -------
        sim: float
            Similarity.
        """
        cdef float sim = -float("inf")
        cdef float a = self.cluster_out_weights[node1] * self.cluster_in_weights[node2]
        cdef float b = self.cluster_out_weights[node2] * self.cluster_in_weights[node1]
        cdef float den = a + b

        if den > 0:
            sim = 2 * self.neighbors[node1][node2] / den
        return sim

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef AggregateGraph merge(self, int node1, int node2):
        """Merges two nodes.

        Parameters
        ----------
        node1, node2 :
            The two nodes to merge.

        Returns
        -------
        self: :class:`AggregateGraph`
            The aggregate grate (without self-loop).
        """
        cdef int new_node = self.next_cluster
        self.neighbors[new_node] = {}
        self.neighbors[new_node][new_node] = 0
        cdef set common_neighbors = set(self.neighbors[node1].keys()) & set(self.neighbors[node2].keys()) - {node1, node2}
        for node in common_neighbors:
            self.neighbors[new_node][node] = self.neighbors[node1].pop(node) + self.neighbors[node2].pop(node)
            self.neighbors[node][new_node] = self.neighbors[node].pop(node1) + self.neighbors[node].pop(node2)
        for node in {node1, node2}:
            for neighbor in set(self.neighbors[node].keys()) - {node1, node2}:
                self.neighbors[new_node][neighbor] = self.neighbors[node].pop(neighbor)
                self.neighbors[neighbor][new_node] = self.neighbors[neighbor].pop(node)
            for other_node in {node1, node2}:
                if other_node in self.neighbors[node]:
                    self.neighbors[new_node][new_node] += self.neighbors[node][other_node]
            del self.neighbors[node]
        self.cluster_sizes[new_node] = self.cluster_sizes.pop(node1) + self.cluster_sizes.pop(node2)
        self.cluster_out_weights[new_node] = self.cluster_out_weights.pop(node1) + self.cluster_out_weights.pop(node2)
        self.cluster_in_weights[new_node] = self.cluster_in_weights.pop(node1) + self.cluster_in_weights.pop(node2)
        self.next_cluster += 1
        return self


class Paris(BaseHierarchy):
    """Agglomerative clustering algorithm that performs greedy merge of nodes based on their similarity.

    The similarity between nodes :math:`i,j` is :math:`\\dfrac{A_{ij}}{w_i w_j}` where

    * :math:`A_{ij}` is the weight of edge :math:`i,j`,
    * :math:`w_i, w_j` are the weights of nodes :math:`i,j`

    If the input matrix :math:`B` is a biadjacency matrix (i.e., rectangular), the algorithm is applied
    to the corresponding adjacency matrix :math:`A  = \\begin{bmatrix} 0 & B \\\\ B^T & 0 \\end{bmatrix}`

    Parameters
    ----------
    weights : str
        Weights of nodes.
        ``'degree'`` (default) or ``'uniform'``.
    reorder : bool
        If ``True`` (default), reorder the dendrogram in non-decreasing order of height.

    Attributes
    ----------
    dendrogram_ : np.ndarray
        Dendrogram of the graph.

    Examples
    --------
    >>> from sknetwork.hierarchy import Paris
    >>> from sknetwork.data import house
    >>> paris = Paris()
    >>> adjacency = house()
    >>> dendrogram = paris.fit_predict(adjacency)
    >>> np.round(dendrogram, 2)
    array([[3.        , 2.        , 0.17      , 2.        ],
           [1.        , 0.        , 0.25      , 2.        ],
           [6.        , 4.        , 0.31      , 3.        ],
           [7.        , 5.        , 0.67      , 5.        ]])

    Notes
    -----
    Each row of the dendrogram = :math:`i, j`, distance, size of cluster :math:`i + j`.


    See Also
    --------
    scipy.cluster.hierarchy.linkage

    References
    ----------
    T. Bonald, B. Charpentier, A. Galland, A. Hollocou (2018).
    `Hierarchical Graph Clustering using Node Pair Sampling.
    <https://arxiv.org/abs/1806.01664>`_
    Workshop on Mining and Learning with Graphs.
    """
    def __init__(self, weights: str = 'degree', reorder: bool = True):
        super(Paris, self).__init__()
        self.dendrogram_ = None
        self.weights = weights
        self.reorder = reorder
        self.bipartite = None

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def fit(self, input_matrix: Union[sparse.csr_matrix, np.ndarray], force_bipartite: bool = False) -> 'Paris':
        """Agglomerative clustering using the nearest neighbor chain.

        Parameters
        ----------
        input_matrix : sparse.csr_matrix, np.ndarray
            Adjacency matrix or biadjacency matrix of the graph.
        force_bipartite :
            If ``True``, force the input matrix to be considered as a biadjacency matrix.

        Returns
        -------
        self: :class:`Paris`
        """
        self._init_vars()

        # input
        adjacency, self.bipartite = get_adjacency(input_matrix, force_bipartite=force_bipartite)

        weights = self.weights
        out_weights = get_probs(weights, adjacency)
        in_weights = get_probs(weights, adjacency.T)

        if not is_symmetric(adjacency):
            adjacency = directed2undirected(adjacency)

        null_weights = (out_weights + in_weights) == 0
        if any(null_weights):
            adjacency += sparse.diags(null_weights.astype(int))

        if adjacency.shape[0] <= 1:
            raise ValueError('The graph must contain at least two nodes.')

        # agglomerative clustering
        aggregate_graph = AggregateGraph(out_weights, in_weights, adjacency.data.astype(float),
                                         adjacency.indices, adjacency.indptr)

        cdef vector[(int, int)] connected_components
        dendrogram = []
        cdef int node
        cdef int next_node
        cdef int cluster_size
        cdef int next_cluster_size
        cdef int neighbor
        cdef int nearest_neighbor
        cdef int nearest_neighbor_last
        cdef vector[int] chain
        cdef float sim
        cdef float max_sim

        while len(aggregate_graph.cluster_sizes):
            for node in aggregate_graph.cluster_sizes:
                break
            chain.clear()
            chain.push_back(node)
            while chain.size():
                node = chain[chain.size() - 1]
                chain.pop_back()
                if set(aggregate_graph.neighbors[node].keys()) - {node}:
                    max_sim = -float("inf")
                    for neighbor in set(aggregate_graph.neighbors[node].keys()) - {node}:
                        sim = aggregate_graph.similarity(node, neighbor)
                        if sim > max_sim:
                            nearest_neighbor = neighbor
                            max_sim = sim
                        elif sim == max_sim:
                            nearest_neighbor = min(neighbor, nearest_neighbor)
                    if chain.size():
                        nearest_neighbor_last = chain[chain.size() - 1]
                        chain.pop_back()
                        if nearest_neighbor_last == nearest_neighbor:
                            size = aggregate_graph.cluster_sizes[node] + aggregate_graph.cluster_sizes[nearest_neighbor]
                            dendrogram.append([node, nearest_neighbor, 1. / max_sim, size])
                            aggregate_graph.merge(node, nearest_neighbor)
                        else:
                            chain.push_back(nearest_neighbor_last)
                            chain.push_back(node)
                            chain.push_back(nearest_neighbor)
                    else:
                        chain.push_back(node)
                        chain.push_back(nearest_neighbor)
                else:
                    connected_components.push_back((node, aggregate_graph.cluster_sizes[node]))
                    del aggregate_graph.cluster_sizes[node]

        node, cluster_size = connected_components[connected_components.size() - 1]
        connected_components.pop_back()
        for next_node, next_cluster_size in connected_components:
            cluster_size += next_cluster_size
            dendrogram.append([node, next_node, float("inf"), cluster_size])
            node = aggregate_graph.next_cluster
            aggregate_graph.next_cluster += 1

        dendrogram = np.array(dendrogram)
        if self.reorder:
            dendrogram = reorder_dendrogram(dendrogram)

        self.dendrogram_ = dendrogram
        if self.bipartite:
            self._split_vars(input_matrix.shape)

        return self
