.. _clustering:

Clustering
**********

Clustering algorithms.

The attribute ``labels_``  assigns a label (cluster index) to each node of the graph.


Louvain
-------
.. autoclass:: sknetwork.clustering.Louvain

.. autoclass:: sknetwork.clustering.BiLouvain


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

