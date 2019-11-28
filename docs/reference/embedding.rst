.. _embedding:

Embedding
*********

.. currentmodule:: sknetwork

This module contains graph embedding algorithms.

The attribute ``.embedding_`` assigns a vector to each node of the graph.

Spectral
--------

.. autoclass:: sknetwork.embedding.Spectral
    :inherited-members:
    :members:

BiSpectral
----------

.. autoclass:: sknetwork.embedding.BiSpectral
    :inherited-members:
    :members:

Metrics
-------
.. autofunction:: sknetwork.embedding.cosine_modularity

