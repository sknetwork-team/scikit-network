.. _clustering:

Clustering
**********

.. currentmodule:: sknetwork

This module contains clustering algorithms.

The attribute ``.labels_``  assigns a label (cluster index) to each node of the graph.

Louvain
-------
.. autoclass:: sknetwork.clustering.Louvain
    :members:

.. autoclass:: sknetwork.clustering.GreedyModularity
    :members:

Spectral
--------
.. autoclass:: sknetwork.clustering.SpectralClustering
    :members:

Metrics
-------
.. autofunction:: sknetwork.clustering.modularity

Post-processing
---------------
.. autofunction:: sknetwork.clustering.post_processing.membership_matrix

.. autofunction:: sknetwork.clustering.post_processing.reindex_clusters

