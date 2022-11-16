.. _hierarchy:

Hierarchy
*********

Hierarchical clustering algorithms.

The attribute ``dendrogram_`` gives the dendrogram.

A dendrogram is an array of size :math:`(n-1) \times 4` representing the successive merges of nodes.
Each row gives the two merged nodes, their distance and the size of the resulting cluster.
Any new node resulting from a merge takes the first available index
(e.g., the first merge corresponds to node :math:`n`).

Paris
-----
.. autoclass:: sknetwork.hierarchy.Paris

Louvain
-------
.. autoclass:: sknetwork.hierarchy.LouvainHierarchy

.. autoclass:: sknetwork.hierarchy.LouvainIteration

Ward
----
.. autoclass:: sknetwork.hierarchy.Ward

Metrics
-------
.. autofunction:: sknetwork.hierarchy.dasgupta_cost

.. autofunction:: sknetwork.hierarchy.dasgupta_score

.. autofunction:: sknetwork.hierarchy.tree_sampling_divergence

Cuts
----
.. autofunction:: sknetwork.hierarchy.cut_straight

.. autofunction:: sknetwork.hierarchy.cut_balanced

Dendrograms
-----------
.. autofunction:: sknetwork.hierarchy.aggregate_dendrogram

.. autofunction:: sknetwork.hierarchy.reorder_dendrogram
