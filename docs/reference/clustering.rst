.. _clustering:

Clustering
**********

Clustering algorithms.

The attribute ``labels_``  assigns a label (cluster index) to each node of the graph.


Louvain
-------

Here are the available notions of modularity for the Louvain algorithm:

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

For bipartite graphs, the considered adjacency matrix is

:math:`A = \begin{pmatrix} 0 & B\\B^T & 0\end{pmatrix}`

for Newman modularity and Potts modularity (i.e., the graph is considered as undirected), and

:math:`A = \begin{pmatrix} 0 & B\\0 & 0\end{pmatrix}`

for Dugué modularity (i.e., the graph is considered as directed).
The latter is the default option and corresponds to Barber's modularity:

:math:`Q = \frac{1}{w} \sum_{i,j}\left(B_{ij} - \gamma \frac{d_if_j}{w}\right)\delta_{c_i,c_j}`

where :math:`i` in the row index, :math:`j` in the column index, :math:`d_i` is the degree of row :math:`i`,
:math:`f_j` is the degree of column :math:`j` and :math:`w = 1^TB1` is the sum of degrees (either rows or columns).

When the graph is weighted, the degree of a node is replaced by its weight (sum of edge weights).

.. autoclass:: sknetwork.clustering.Louvain

Propagation
-----------
.. autoclass:: sknetwork.clustering.PropagationClustering

K-Means
-------
.. autoclass:: sknetwork.clustering.KMeans

Post-processing
---------------
.. autofunction:: sknetwork.clustering.reindex_labels

.. autofunction:: sknetwork.clustering.aggregate_graph

Metrics
-------
.. autofunction:: sknetwork.clustering.get_modularity

