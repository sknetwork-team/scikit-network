.. _clustering:

Clustering
**********

.. currentmodule:: sknetwork

Louvain
-------
.. automodule:: sknetwork.clustering

.. autoclass:: sknetwork.clustering.louvain.Louvain
    :members:

.. autoclass:: sknetwork.clustering.louvain.GreedyModularity
    :members:

.. autoclass:: sknetwork.clustering.louvain.GreedyModularityNumba
    :members:

Louvain for bipartite graphs
----------------------------

.. autoclass:: sknetwork.clustering.bilouvain.BiLouvain
    :members:

.. autoclass:: sknetwork.clustering.bilouvain.GreedyBipartite
    :members:

.. autoclass:: sknetwork.clustering.bilouvain.GreedyBipartiteNumba
    :members:


Metrics
-------
.. autofunction:: sknetwork.clustering.metrics.modularity

.. autofunction:: sknetwork.clustering.metrics.bimodularity

.. autofunction:: sknetwork.clustering.metrics.cocitation_modularity

.. autofunction:: sknetwork.clustering.metrics.performance

.. autofunction:: sknetwork.clustering.metrics.cocitation_performance
