.. _embedding:

Embedding
*********

.. currentmodule:: sknetwork

This submodule contains embedding algorithms, characterized by their ``.embedding_`` attribute which assigns a dense
vector representation to each node in the graph.


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



