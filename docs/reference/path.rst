.. _path:

Path
****

Standard algorithms related to graph traversal.

Most algorithms are adapted from SciPy_.

Shortest path
-------------

.. autofunction:: sknetwork.path.get_distances

.. autofunction:: sknetwork.path.get_shortest_path

Summary of the different methods and their worst-case complexity for :math:`n` nodes and :math:`m` edges (for the
all-pairs problem):

+----------------+------------------------------+----------------------------------------+
|     Method     |  Worst-case time complexity  |                 Remarks                |
+================+==============================+========================================+
|    Dijkstra    | :math:`O(n^2 \log n + nm)`   | For use on graphs                      |
|                |                              | with positive weights only             |
+----------------+------------------------------+----------------------------------------+
|  Bellman-Ford  | :math:`O(nm)`                |                                        |
+----------------+------------------------------+ For use on graphs without              |
|    Johnson     | :math:`O(n^2 \log n + nm)`   | negative-weight cycles only            |
+----------------+------------------------------+----------------------------------------+


Search
------

.. autofunction:: sknetwork.path.breadth_first_search

.. autofunction:: sknetwork.path.depth_first_search

Metrics
-------

.. autofunction:: sknetwork.path.get_diameter
.. autofunction:: sknetwork.path.get_radius
.. autofunction:: sknetwork.path.get_eccentricity




.. _SciPy: https://docs.scipy.org/doc/scipy/reference/sparse.csgraph.html
