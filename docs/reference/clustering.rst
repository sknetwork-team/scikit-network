.. _clustering:

Clustering
**********

Clustering algorithms.

The attribute ``labels_``  assigns a label (cluster index) to each node of the graph.


Louvain
-------

Here are the available modularities for the Louvain algorithm:

+------------------------------+--------------------------------------------------------------------------------------------------------------+-------------------------+
| Modularity                   | Formula                                                                                                      | Remarks                 |
+==============================+==============================================================================================================+=========================+
| Newman (``'newman'``)        | :math:`Q = \dfrac{1}{w} \sum_{i,j}\left(A_{ij} - \gamma \dfrac{d_id_j}{w}\right)\delta_{c_i,c_j}`            | Ignores edge directions |
+------------------------------+--------------------------------------------------------------------------------------------------------------+-------------------------+
| Dugué (``'dugue'``)          | :math:`Q = \dfrac{1}{w} \sum_{i,j}\left(A_{ij} - \gamma \dfrac{d^+_id^-_j}{w}\right)\delta_{c_i,c_j}`        |                         |
+------------------------------+--------------------------------------------------------------------------------------------------------------+-------------------------+
| Constant Potts (``'potts'``) | :math:`Q =  \sum_{i,j}\left(\dfrac{A_{ij}}{w} - \gamma \dfrac{1}{n^2}\right)\delta_{c_i,c_j}`                | Ignores edge directions |
+------------------------------+--------------------------------------------------------------------------------------------------------------+-------------------------+


.. autoclass:: sknetwork.clustering.Louvain

.. autoclass:: sknetwork.clustering.BiLouvain

Propagation
-----------
.. autoclass:: sknetwork.clustering.PropagationClustering

.. autoclass:: sknetwork.clustering.BiPropagationClustering

K-Means
-------
.. autoclass:: sknetwork.clustering.KMeans

.. autoclass:: sknetwork.clustering.BiKMeans


Metrics
-------
.. autofunction:: sknetwork.clustering.modularity

.. autofunction:: sknetwork.clustering.bimodularity

.. autofunction:: sknetwork.clustering.comodularity

.. autofunction:: sknetwork.clustering.normalized_std

Post-processing
---------------

.. autofunction:: sknetwork.clustering.postprocess.reindex_labels

