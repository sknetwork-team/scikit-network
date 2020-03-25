.. _hierarchy:

Hierarchy
*********

.. currentmodule:: sknetwork

This module contains hierarchical clustering algorithms. The attribute ``.dendrogram_`` contains the dendrogram.

A dendrogram is an array of size :math:`(n-1) \times 4` representing the successive merges of nodes:

* The first two columns contain the indices of the merges nodes.

* The third column gives the distance between these nodes.

* The last column gives the size of the corresponding cluster (in number of nodes) after the merge.

Any new node resulting from a merge takes the first available index (e.g., the first merge corresponds to node :math:`n`).


Paris
-----
.. autoclass:: sknetwork.hierarchy.Paris
    :inherited-members:
    :members:

.. autoclass:: sknetwork.hierarchy.BiParis
    :inherited-members:
    :members:

Ward
----
.. autoclass:: sknetwork.hierarchy.Ward
    :inherited-members:
    :members:

.. autoclass:: sknetwork.hierarchy.BiWard
    :inherited-members:
    :members:


Cuts
----
.. autofunction:: sknetwork.hierarchy.cut_straight

.. autofunction:: sknetwork.hierarchy.cut_balanced


Metrics
-------
.. autofunction:: sknetwork.hierarchy.dasgupta_score

.. autofunction:: sknetwork.hierarchy.tree_sampling_divergence

