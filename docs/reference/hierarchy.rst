.. _hierarchical_clustering:

Hierarchical clustering
***********************

.. currentmodule:: sknetwork

This submodule contains embedding algorithms, characterized by their ``.dendrogram_`` attribute.

A dendrogram is an :math:`(n-1) \times 4` array ``Z`` representing the successive merges of clusters i.e.
clsuters with indices ``Z[i, 0]`` and ``Z[i, 1]`` which are at distance ``Z[i, 2]`` are merged into cluster :math:`n+i`
which contains ``Z[i, 3]`` samples.


Paris
-----
.. automodule:: sknetwork.hierarchy
.. autosummary::
   :toctree: generated/

.. autoclass:: sknetwork.hierarchy.Paris
    :members:


Utils
-----
.. autofunction:: sknetwork.hierarchy.reorder_dendrogram

.. autofunction:: sknetwork.hierarchy.cut



Metrics
-------
.. autofunction:: sknetwork.hierarchy.dasgupta_cost

.. autofunction:: sknetwork.hierarchy.tree_sampling_divergence

