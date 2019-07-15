.. _clustering:

Clustering
**********

.. currentmodule:: sknetwork

This module contains clustering algorithms. The attribute ``.labels_``  assigns a cluster to each node of the graph.

Louvain
-------
.. automodule:: sknetwork.clustering

.. autoclass:: sknetwork.clustering.Louvain
    :members:

.. autoclass:: sknetwork.clustering.GreedyModularity
    :members:


Louvain for bipartite graphs
----------------------------

.. autoclass:: sknetwork.clustering.BiLouvain
    :members:


Metrics
-------
.. autofunction:: sknetwork.clustering.modularity

.. autofunction:: sknetwork.clustering.bimodularity

.. autofunction:: sknetwork.clustering.cocitation_modularity
