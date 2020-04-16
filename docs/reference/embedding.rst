.. _embedding:

Embedding
*********

.. currentmodule:: sknetwork

This module contains graph embedding algorithms.

The attribute ``embedding_`` assigns a vector to each node of the graph.

Spectral
--------

.. autoclass:: sknetwork.embedding.Spectral

.. autoclass:: sknetwork.embedding.BiSpectral

SVD
---

.. autoclass:: sknetwork.embedding.SVD

GSVD
----

.. autoclass:: sknetwork.embedding.GSVD

Layouts
-------

.. autoclass:: sknetwork.embedding.FruchtermanReingold

Metrics
-------
.. autofunction:: sknetwork.embedding.cosine_modularity

