.. _hierarchy:

Hierarchical clustering
***********************

.. currentmodule:: sknetwork

This module contains hierarchical clustering algorithms. The attribute ``.dendrogram_`` contains the dendrogram.

A dendrogram is an :math:`(n-1) \times 4` array ``Z`` representing the successive merges of clusters, i.e.,
clusters of indices ``Z[i, 0]`` and ``Z[i, 1]``, which are at distance ``Z[i, 2]``, are merged into cluster of index
:math:`n+i`, which contains ``Z[i, 3]`` nodes.


Paris
-----
.. automodule:: sknetwork.hierarchy

.. autoclass:: sknetwork.hierarchy.Paris
    :members:


Cuts
----
.. autofunction:: sknetwork.hierarchy.straight_cut



Metrics
-------
.. autofunction:: sknetwork.hierarchy.dasgupta_cost

.. autofunction:: sknetwork.hierarchy.tree_sampling_divergence

