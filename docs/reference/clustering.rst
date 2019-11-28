.. _clustering:

Clustering
**********

.. currentmodule:: sknetwork

This module contains clustering algorithms.

The attribute ``.labels_``  assigns a label (cluster index) to each node of the graph.


Louvain
-------
.. autoclass:: sknetwork.clustering.Louvain
    :inherited-members:
    :members:

.. autoclass:: sknetwork.clustering.BiLouvain
    :inherited-members:
    :show-inheritance:
    :members:

.. autoclass:: sknetwork.clustering.GreedyModularity
    :inherited-members:
    :members:

Spectral
--------
.. autoclass:: sknetwork.clustering.SpectralClustering
    :inherited-members:
    :members:

.. autoclass:: sknetwork.clustering.BiSpectralClustering
    :inherited-members:
    :members:

Metrics
-------
.. autofunction:: sknetwork.clustering.modularity

.. autofunction:: sknetwork.clustering.bimodularity

.. autofunction:: sknetwork.clustering.cocitation_modularity

.. autofunction:: sknetwork.clustering.nsd

Post-processing
---------------
.. autofunction:: sknetwork.clustering.post_processing.membership_matrix

.. autofunction:: sknetwork.clustering.post_processing.reindex_clusters

