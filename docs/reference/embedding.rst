.. _embedding:

Embedding
*********

Graph embedding algorithms.

The attribute ``embedding_`` assigns a vector to each node of the graph.

Spectral
--------

.. autoclass:: sknetwork.embedding.Spectral

SVD
---

.. autoclass:: sknetwork.embedding.SVD

GSVD
----

.. autoclass:: sknetwork.embedding.GSVD

PCA
---

.. autoclass:: sknetwork.embedding.PCA

Random Projection
-----------------

.. autoclass:: sknetwork.embedding.RandomProjection

Louvain
-------

.. autoclass:: sknetwork.embedding.LouvainEmbedding

Hierarchical Louvain
--------------------

.. autoclass:: sknetwork.embedding.LouvainNE

Force Atlas
-----------

.. autoclass:: sknetwork.embedding.ForceAtlas

Spring
------

.. autoclass:: sknetwork.embedding.Spring

Metrics
-------
.. autofunction:: sknetwork.embedding.get_cosine_similarity

