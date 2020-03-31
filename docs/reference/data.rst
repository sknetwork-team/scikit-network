.. _data:

Data
####

Sources and tools for importing data.

Graphs
******

.. currentmodule:: sknetwork

.. autoclass:: sknetwork.data.BaseGraph
    :inherited-members:
    :members:

Toy graphs
**********

Synthetic data
--------------

.. autoclass:: sknetwork.data.House
    :inherited-members:
    :members:

.. autoclass:: sknetwork.data.BowTie
    :inherited-members:
    :members:

Real-world data
---------------

.. autoclass:: sknetwork.data.KarateClub
    :inherited-members:
    :members:

.. autoclass:: sknetwork.data.Miserables
    :inherited-members:
    :members:

.. autoclass:: sknetwork.data.Painters
    :inherited-members:
    :members:

.. autoclass:: sknetwork.data.StarWars
    :inherited-members:
    :members:

.. autoclass:: sknetwork.data.MovieActor
    :inherited-members:
    :members:

Random graphs
-------------

.. autoclass:: sknetwork.data.Line
    :inherited-members:
    :members:

.. autoclass:: sknetwork.data.LineDirected
    :inherited-members:
    :members:

.. autoclass:: sknetwork.data.Cycle
    :inherited-members:
    :members:

.. autoclass:: sknetwork.data.CycleDirected
    :inherited-members:
    :members:

.. autofunction:: sknetwork.data.block_model

Loading
*******

.. autofunction:: sknetwork.data.load_wikilinks

.. autofunction:: sknetwork.data.load_konect

Parsing
*******

You can find some datasets on NetRep_.

.. currentmodule:: sknetwork

.. autofunction:: sknetwork.data.parse_tsv

.. autofunction:: sknetwork.data.parse_labels

.. _NetRep: http://networkrepository.com/

