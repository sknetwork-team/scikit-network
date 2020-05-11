.. _ranking:

Ranking
*******

Node ranking algorithms.

The attribute ``scores_`` assigns a score of importance to each node of the graph.

PageRank
--------
.. autoclass:: sknetwork.ranking.PageRank

.. autoclass:: sknetwork.ranking.BiPageRank

.. autoclass:: sknetwork.ranking.CoPageRank

Diffusion
---------
.. autoclass:: sknetwork.ranking.Diffusion

.. autoclass:: sknetwork.ranking.BiDiffusion

Katz
----
.. autoclass:: sknetwork.ranking.Katz

.. autoclass:: sknetwork.ranking.BiKatz

.. autoclass:: sknetwork.ranking.CoKatz

HITS
----
.. autoclass:: sknetwork.ranking.HITS

Closeness centrality
--------------------
.. autoclass:: sknetwork.ranking.Closeness

Harmonic centrality
-------------------
.. autoclass:: sknetwork.ranking.Harmonic

Post-processing
---------------

.. autofunction:: sknetwork.ranking.top_k
