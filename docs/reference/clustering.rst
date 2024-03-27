.. _clustering:

Clustering
**********

Clustering algorithms.

The attribute ``labels_``  assigns a label (cluster index) to each node of the graph.


Louvain
-------

The Louvain algorithm aims at maximizing the modularity. Several variants of modularity are available:

+-------------------------+------------------------------------------------------------------------------------------------------+
| Modularity              | Formula                                                                                              |
+=========================+======================================================================================================+
| Newman (``'newman'``)   | :math:`Q = \frac{1}{w} \sum_{i,j}\left(A_{ij} - \gamma \frac{d_id_j}{w}\right)\delta_{c_i,c_j}`      |
+-------------------------+------------------------------------------------------------------------------------------------------+
| Dugué (``'dugue'``)     | :math:`Q = \frac{1}{w} \sum_{i,j}\left(A_{ij} - \gamma \frac{d^+_id^-_j}{w}\right)\delta_{c_i,c_j}`  |
+-------------------------+------------------------------------------------------------------------------------------------------+
| Potts (``'potts'``)     | :math:`Q =  \sum_{i,j}\left(\frac{A_{ij}}{w} - \gamma \frac{1}{n^2}\right)\delta_{c_i,c_j}`          |
+-------------------------+------------------------------------------------------------------------------------------------------+

where
    * :math:`A` is the adjacency matrix,
    * :math:`c_i` is the cluster of node :math:`i`,
    * :math:`d_i` is the degree of node :math:`i`,
    * :math:`d^+_i, d^-_i` are the out-degree, in-degree of node :math:`i` (for directed graphs),
    * :math:`w = 1^TA1` is the sum of degrees,
    * :math:`\delta` is the Kronecker symbol,
    * :math:`\gamma \ge 0` is the resolution parameter.

Observe that for undirected graphs, the Newman and Dugué variants are equivalent.

For bipartite graphs, the considered adjacency matrix :math:`A` depends on the modularity.
For the Newman and Potts variants, the graph is considered as undirected so that:

:math:`A = \begin{pmatrix} 0 & B\\B^T & 0\end{pmatrix}`

For the Dugué variant, the graph is considered as directed so that:

:math:`A = \begin{pmatrix} 0 & B\\0 & 0\end{pmatrix}`

This is the default option and corresponds to Barber's modularity (see reference below).

When the graph is weighted, the degree of a node is replaced by its weight (sum of edge weights).

.. autoclass:: sknetwork.clustering.Louvain

Leiden
------

.. autoclass:: sknetwork.clustering.Leiden

K-centers
---------

.. autoclass:: sknetwork.clustering.KCenters


Propagation
-----------
.. autoclass:: sknetwork.clustering.PropagationClustering

Post-processing
---------------
.. autofunction:: sknetwork.clustering.reindex_labels

.. autofunction:: sknetwork.clustering.aggregate_graph

Metrics
-------
.. autofunction:: sknetwork.clustering.get_modularity

