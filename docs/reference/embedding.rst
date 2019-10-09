.. _embedding:

Embedding
*********

.. currentmodule:: sknetwork

This module contains graph embedding algorithms.

The attribute ``.embedding_`` assigns a vector to each node of the graph.


Spectral
--------

.. autoclass:: sknetwork.embedding.Spectral
    :members:

SVD
---

.. autoclass:: sknetwork.embedding.SVD
    :members:

Metrics
-------
.. autofunction:: sknetwork.embedding.dot_modularity

.. autofunction:: sknetwork.embedding.hscore



