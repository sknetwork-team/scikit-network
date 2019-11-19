.. _ranking:

Ranking
*******

.. currentmodule:: sknetwork

This module contains ranking algorithms. The attribute ``.score_`` assigns an importance
score to each node of the graph.

PageRank
--------
.. autoclass:: sknetwork.ranking.PageRank
    :members:

.. autoclass:: sknetwork.ranking.BiPageRank
    :show-inheritance:
    :inherited-members:
    :members:

Diffusion
---------
.. autoclass:: sknetwork.ranking.Diffusion
    :members:

.. autoclass:: sknetwork.ranking.BiDiffusion
    :members:

HITS
----
.. autoclass:: sknetwork.ranking.HITS
    :members:

Closeness centrality
--------------------
.. autoclass:: sknetwork.ranking.Closeness
    :members:

Harmonic centrality
-------------------
.. autoclass:: sknetwork.ranking.Harmonic
    :members:
