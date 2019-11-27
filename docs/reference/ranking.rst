.. _ranking:

Ranking
*******

.. currentmodule:: sknetwork

This module contains ranking algorithms. The attribute ``.scores_`` assigns an importance
score to each node of the graph.

PageRank
--------
.. autoclass:: sknetwork.ranking.PageRank
    :inherited-members:
    :members:

.. autoclass:: sknetwork.ranking.BiPageRank
    :show-inheritance:
    :inherited-members:
    :members:

Diffusion
---------
.. autoclass:: sknetwork.ranking.Diffusion
    :inherited-members:
    :members:

.. autoclass:: sknetwork.ranking.BiDiffusion
    :inherited-members:
    :members:

HITS
----
.. autoclass:: sknetwork.ranking.HITS
    :inherited-members:
    :members:

Closeness centrality
--------------------
.. autoclass:: sknetwork.ranking.Closeness
    :inherited-members:
    :members:

Harmonic centrality
-------------------
.. autoclass:: sknetwork.ranking.Harmonic
    :inherited-members:
    :members:
