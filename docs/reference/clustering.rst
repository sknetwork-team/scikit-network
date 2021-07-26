.. _clustering:

Clustering
**********

Clustering algorithms.

The attribute ``labels_``  assigns a label (cluster index) to each node of the graph.


Louvain
-------

Here are the available notions of modularity for the Louvain algorithm:

+------------------------------+--------------------------------------------------------------------------------------------------------------+
| Modularity                   | Formula                                                                                                      |
+==============================+==============================================================================================================+
| Newman (``'newman'``)        | :math:`Q = \dfrac{1}{w} \sum_{i,j}\left(A_{ij} - \gamma \dfrac{d_id_j}{w}\right)\delta_{c_i,c_j}`            |
+------------------------------+--------------------------------------------------------------------------------------------------------------+
| Dugué (``'dugue'``)          | :math:`Q = \dfrac{1}{w} \sum_{i,j}\left(A_{ij} - \gamma \dfrac{d^+_id^-_j}{w}\right)\delta_{c_i,c_j}`        |
+------------------------------+--------------------------------------------------------------------------------------------------------------+
| Constant Potts (``'potts'``) | :math:`Q =  \sum_{i,j}\left(\dfrac{A_{ij}}{w} - \gamma \dfrac{1}{n^2}\right)\delta_{c_i,c_j}`                |
+------------------------------+--------------------------------------------------------------------------------------------------------------+

where
    * :math:`c_i` is the cluster of node :math:`i`,
    * :math:`d_i` is the weight of node :math:`i`,
    * :math:`d^+_i, d^-_i` are the out-weight, in-weight of node :math:`i` (for directed graphs),
    * :math:`w = 1^TA1` is the total weight of the graph,
    * :math:`\\delta` is the Kronecker symbol,
    * :math:`\\gamma \\ge 0` is the resolution parameter.

The adjacency matrix :math:`A` is made symmetric if asymmetric.

.. autoclass:: sknetwork.clustering.Louvain

Propagation
-----------
.. autoclass:: sknetwork.clustering.PropagationClustering

K-Means
-------
.. autoclass:: sknetwork.clustering.KMeans

Metrics
-------
.. autofunction:: sknetwork.clustering.modularity

.. autofunction:: sknetwork.clustering.bimodularity

.. autofunction:: sknetwork.clustering.comodularity

.. autofunction:: sknetwork.clustering.normalized_std

Post-processing
---------------

.. autofunction:: sknetwork.clustering.postprocess.reindex_labels

