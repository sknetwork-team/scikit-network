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

.. autoclass:: sknetwork.clustering.BiLouvain
    :members:

.. autoclass:: sknetwork.clustering.GreedyModularity
    :members:

Spectral
--------
.. autoclass:: sknetwork.clustering.SpectralClustering
    :members:

.. autoclass:: sknetwork.clustering.BiSpectralClustering
    :members:

Metrics
-------
.. autofunction:: sknetwork.clustering.modularity

.. autofunction:: sknetwork.clustering.bimodularity

.. autofunction:: sknetwork.clustering.cocitation_modularity

Post-processing
---------------
.. autofunction:: sknetwork.clustering.post_processing.membership_matrix

.. autofunction:: sknetwork.clustering.post_processing.reindex_clusters

