.. _embedding:

Embedding
*********

.. currentmodule:: sknetwork

This module contains embedding algorithms. The attribute ``.embedding_`` assigns a vector to each node of the graph.


Spectral
--------
.. automodule:: sknetwork.embedding

.. autoclass:: sknetwork.embedding.Spectral
    :members:

SVD
---

.. autoclass:: sknetwork.embedding.GSVD
    :members:

Metrics
-------
.. autofunction:: sknetwork.embedding.dot_modularity

.. autofunction:: sknetwork.embedding.hscore



