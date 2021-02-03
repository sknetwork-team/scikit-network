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

PCA
---

.. autoclass:: sknetwork.embedding.PCA

Random Projection
-----------------

.. autoclass:: sknetwork.embedding.RandomProjection

.. autoclass:: sknetwork.embedding.BiRandomProjection

Louvain
-------

.. autoclass:: sknetwork.embedding.LouvainEmbedding

.. autoclass:: sknetwork.embedding.BiLouvainEmbedding

Hierarchical Louvain
--------------------

.. autoclass:: sknetwork.embedding.HLouvainEmbedding

.. autoclass:: sknetwork.embedding.BiHLouvainEmbedding

Force Atlas
-----------

.. autoclass:: sknetwork.embedding.ForceAtlas

Spring
------

.. autoclass:: sknetwork.embedding.Spring

Metrics
-------
.. autofunction:: sknetwork.embedding.cosine_modularity

