.. _clustering:

Clustering
**********

.. currentmodule:: sknetwork

This module contains clustering algorithms.

The attribute ``.labels_``  assigns a label (cluster index) to each node of the graph.

Louvain
-------
.. automodule:: sknetwork.clustering

.. autoclass:: sknetwork.clustering.Louvain
    :members:

.. autoclass:: sknetwork.clustering.GreedyModularity
    :members:

Metrics
-------
.. autofunction:: sknetwork.clustering.modularity

.. autofunction:: sknetwork.clustering.cocitation_modularity
