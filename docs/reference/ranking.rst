.. _ranking:

Ranking
*******

Node ranking algorithms.

The attribute ``scores_`` assigns a score of importance to each node of the graph.

PageRank
--------
.. autoclass:: sknetwork.ranking.PageRank

Katz
----
.. autoclass:: sknetwork.ranking.Katz

HITS
----
.. autoclass:: sknetwork.ranking.HITS

Betweenness centrality
----------------------
.. autoclass:: sknetwork.ranking.Betweenness

Closeness centrality
--------------------
.. autoclass:: sknetwork.ranking.Closeness

Harmonic centrality
-------------------
.. autoclass:: sknetwork.ranking.Harmonic

Post-processing
---------------
.. autofunction:: sknetwork.ranking.top_k
