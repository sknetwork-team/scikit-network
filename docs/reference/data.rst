.. _data:

Data
####

Sources and tools for importing synthetic and real data.

Toy graphs
**********

.. currentmodule:: sknetwork

Synthetic data
--------------

Undirected graphs
=================

.. autofunction:: sknetwork.data.bow_tie

.. autofunction:: sknetwork.data.house

.. autofunction:: sknetwork.data.simple_directed_graph

Directed graphs
===============

.. autofunction:: sknetwork.data.rock_paper_scissors

.. autofunction:: sknetwork.data.line_graph

Bipartite graphs
================

.. autofunction:: sknetwork.data.simple_bipartite_graph

Real-world data
---------------

Undirected graphs
=================

.. autofunction:: sknetwork.data.karate_club

.. autofunction:: sknetwork.data.miserables

Directed graphs
===============

.. autofunction:: sknetwork.data.painters

Bipartite graphs
================

.. autofunction:: sknetwork.data.movie_actor

.. autofunction:: sknetwork.data.star_wars_villains

Random graph models
-------------------

.. autofunction:: sknetwork.data.block_model

Loading
*******

.. autofunction:: sknetwork.data.load_wikilinks_dataset

.. autofunction:: sknetwork.data.load_konect_dataset

Parsing
*******

You can find some datasets on NetRep_.

.. currentmodule:: sknetwork

.. autofunction:: sknetwork.data.parse_tsv

.. autofunction:: sknetwork.data.parse_labels

.. _NetRep: http://networkrepository.com/

