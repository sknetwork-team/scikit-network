.. _hierarchy:

Hierarchical clustering
***********************

.. currentmodule:: sknetwork

This module contains hierarchical clustering algorithms. The attribute ``.dendrogram_`` contains the dendrogram.

A dendrogram is an array of size :math:`(n-1) \times 4` representing the successive merges of nodes:

* The first two columns contain the indices of the merges nodes.

* The third column gives the distance between these nodes.

* The last column gives the size of the corresponding cluster (in number of nodes) after the merge.

Any new node resulting from a merge takes the first available index (e.g., the first merge corresponds to node :math:`n`).


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

