.. _clustering:

Clustering
**********

.. currentmodule:: sknetwork

This module contains clustering algorithms. The ``.labels_`` attribute assigns a cluster to
each node in the graph.

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
<<<<<<< Updated upstream
=======

>>>>>>> Stashed changes
