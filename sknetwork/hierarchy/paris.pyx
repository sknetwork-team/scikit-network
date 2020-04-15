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

ctypedef np.int_t int_type_t
ctypedef np.float_t float_type_t

from libcpp.vector cimport vector

from typing import Union

from scipy import sparse

from sknetwork.hierarchy.base import BaseHierarchy
from sknetwork.hierarchy.postprocess import reorder_dendrogram, split_dendrogram
from sknetwork.utils.format import bipartite2undirected
from sknetwork.utils.check import check_format, check_probs, is_square


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
    shape :
        Matrix shape.

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
    cdef dict cluster_sizes
    cdef public dict cluster_out_weights
    cdef public dict cluster_in_weights

    def __init__(self, np.float_t[:] out_weights, np.float_t[:] in_weights, np.float_t[:] data, int[:] indices,
                 int[:] indptr, (int, int) shape):
        cdef int n = shape[0]
        cdef float total_weight = sum(data) / 2
        cdef int node

        self.next_cluster = n
        self.neighbors = {}
        for node in range(n):
            # normalize so that the sum of edge weights is equal to 1
            # remove self-loops
            self.neighbors[node] = {indices[i]: data[i] / total_weight for i in
                                    range(indptr[node], indptr[node + 1])
                                    if indices[i] != node}
        self.cluster_sizes = {node: 1 for node in range(n)}
        self.cluster_out_weights = {node: out_weights[node] for node in range(n)}
        self.cluster_in_weights = {node: in_weights[node] for node in range(n)}

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
        cdef set common_neighbors = set(self.neighbors[node1].keys()) & set(self.neighbors[node2].keys()) - {node1, node2}
        for node in common_neighbors:
            self.neighbors[new_node][node] = self.neighbors[node1][node] + self.neighbors[node2][node]
            self.neighbors[node][new_node] = self.neighbors[node].pop(node1) + self.neighbors[node].pop(node2)
        cdef set node1_neighbors = set(self.neighbors[node1].keys()) - set(self.neighbors[node2].keys()) - {node2}
        for node in node1_neighbors:
            self.neighbors[new_node][node] = self.neighbors[node1][node]
            self.neighbors[node][new_node] = self.neighbors[node].pop(node1)
        cdef set node2_neighbors = set(self.neighbors[node2].keys()) - set(self.neighbors[node1].keys()) - {node1}
        for node in node2_neighbors:
            self.neighbors[new_node][node] = self.neighbors[node2][node]
            self.neighbors[node][new_node] = self.neighbors[node].pop(node2)
        del self.neighbors[node1]
        del self.neighbors[node2]
        self.cluster_sizes[new_node] = self.cluster_sizes.pop(node1) + self.cluster_sizes.pop(node2)
        self.cluster_out_weights[new_node] = self.cluster_out_weights.pop(node1) + self.cluster_out_weights.pop(node2)
        self.cluster_in_weights[new_node] = self.cluster_in_weights.pop(node1) + self.cluster_in_weights.pop(node2)
        self.next_cluster += 1
        return self


class Paris(BaseHierarchy):
    """Agglomerative clustering algorithm that performs greedy merge of nodes based on their similarity.

    * Graphs
    * Digraphs

    The similarity between nodes :math:`i,j` is :math:`\\dfrac{A_{ij}}{w_i w_j}` where

    * :math:`A_{ij}` is the weight of edge :math:`i,j`,
    * :math:`w_i, w_j` are the weights of nodes :math:`i,j`

    Parameters
    ----------
    weights :
        Weights of nodes.
        ``'degree'`` (default) or ``'uniform'``.
    reorder :
        If ``True``, reorder the dendrogram in non-decreasing order of height.

    Attributes
    ----------
    dendrogram_ : numpy array of shape (total number of nodes - 1, 4)
        Dendrogram.

    Examples
    --------
    >>> from sknetwork.hierarchy import Paris
    >>> from sknetwork.data import house
    >>> paris = Paris()
    >>> adjacency = house()
    >>> dendrogram = paris.fit_transform(adjacency)
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

        self.weights = weights
        self.reorder = reorder

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'Paris':
        """Agglomerative clustering using the nearest neighbor chain.

        Parameters
        ----------
        adjacency :
            Adjacency matrix of the graph.

        Returns
        -------
        self: :class:`Paris`
        """
        adjacency = check_format(adjacency)
        if not is_square(adjacency):
            raise ValueError('The adjacency matrix is not square. Use BiParis() instead.')
        n = adjacency.shape[0]
        sym_adjacency = adjacency + adjacency.T

        weights = self.weights
        out_weights = check_probs(weights, adjacency)
        in_weights = check_probs(weights, adjacency.T)

        if n <= 1:
            raise ValueError('The graph must contain at least two nodes.')

        aggregate_graph = AggregateGraph(out_weights, in_weights, sym_adjacency.data.astype(np.float), sym_adjacency.indices, sym_adjacency.indptr, sym_adjacency.shape)

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
                if aggregate_graph.neighbors[node]:
                    max_sim = -float("inf")
                    for neighbor in aggregate_graph.neighbors[node]:
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
                            dendrogram.append([node, nearest_neighbor, 1. / max_sim,
                                               aggregate_graph.cluster_sizes[node]
                                               + aggregate_graph.cluster_sizes[nearest_neighbor]])
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
        return self


class BiParis(Paris):
    """Hierarchical clustering of bipartite graphs by the Paris method.

    * Bigraphs

    Parameters
    ----------
    weights :
        Weights of nodes.
        ``'degree'`` (default) or ``'uniform'``.
    reorder :
        If ``True``, reorder the dendrogram in non-decreasing order of height.

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
    >>> from sknetwork.hierarchy import BiParis
    >>> from sknetwork.data import star_wars
    >>> biparis = BiParis()
    >>> biadjacency = star_wars()
    >>> dendrogram = biparis.fit_transform(biadjacency)
    >>> np.round(dendrogram, 2)
    array([[1.        , 2.        , 0.37      , 2.        ],
           [4.        , 0.        , 0.55      , 3.        ],
           [5.        , 3.        , 0.75      , 4.        ]])

    Notes
    -----
    Each row of the dendrogram = :math:`i, j`, height, size of cluster.

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
        Paris.__init__(self, weights, reorder)

        self.dendrogram_row_ = None
        self.dendrogram_col_ = None
        self.dendrogram_full_ = None

    def fit(self, biadjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'BiParis':
        """Apply the Paris algorithm to

        :math:`A  = \\begin{bmatrix} 0 & B \\\\ B^T & 0 \\end{bmatrix}`

        where :math:`B` is the biadjacency matrix of the graph.

        Parameters
        ----------
        biadjacency:
            Biadjacency matrix of the graph.

        Returns
        -------
        self: :class:`BiParis`
        """
        paris = Paris(weights=self.weights)
        biadjacency = check_format(biadjacency)

        adjacency = bipartite2undirected(biadjacency)
        dendrogram = paris.fit_transform(adjacency)
        dendrogram_row, dendrogram_col = split_dendrogram(dendrogram, biadjacency.shape)

        self.dendrogram_ = dendrogram_row
        self.dendrogram_row_ = dendrogram_row
        self.dendrogram_col_ = dendrogram_col
        self.dendrogram_full_ = dendrogram

        return self
