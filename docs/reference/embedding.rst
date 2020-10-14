.. _embedding:

Embedding
*********

Graph embedding algorithms.

The attribute ``embedding_`` assigns a vector to each node of the graph.

Spectral
--------

.. autoclass:: sknetwork.embedding.Spectral

.. autoclass:: sknetwork.embedding.BiSpectral

.. autoclass:: sknetwork.embedding.LaplacianEmbedding

SVD
---

.. autoclass:: sknetwork.embedding.SVD

GSVD
----

.. autoclass:: sknetwork.embedding.GSVD

Louvain
-------

.. autoclass:: sknetwork.embedding.LouvainEmbedding

.. autoclass:: sknetwork.embedding.BiLouvainEmbedding

Force Atlas 2
-------------

.. autoclass:: sknetwork.embedding.ForceAtlas2

Spring
------

.. autoclass:: sknetwork.embedding.Spring

Metrics
-------
.. autofunction:: sknetwork.embedding.cosine_modularity

