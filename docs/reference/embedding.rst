.. _embedding:

Embedding
*********

.. currentmodule:: sknetwork

This module contains graph embedding algorithms.

The attribute ``embedding_`` assigns a vector to each node of the graph.

Spectral
--------

.. autoclass:: sknetwork.embedding.Spectral
    :inherited-members:
    :members:

.. autoclass:: sknetwork.embedding.BiSpectral
    :inherited-members:
    :members:

SVD
---

.. autoclass:: sknetwork.embedding.SVD
    :inherited-members:
    :members:

GSVD
----

.. autoclass:: sknetwork.embedding.GSVD
    :inherited-members:
    :members:

Metrics
-------
.. autofunction:: sknetwork.embedding.cosine_modularity

