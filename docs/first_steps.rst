.. _getting_started:

:mod:`scikit-network` is an open-source python package for the analysis of large graphs.

Each graph is represented by a sparse matrix in ``scipy`` format (CSR).

Installation
------------

To install :mod:`scikit-network`, run this command in your terminal:

.. code-block:: console

    $ pip install scikit-network

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

Alternately, you can download the sources from the `Github repo`_ and run:

.. code-block:: console

    $ cd <scikit-network folder>
    $ python setup.py develop


.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/
.. _Github repo: https://github.com/sknetwork-team/scikit-network

Import
------

Import :mod:`scikit-network` in Python:

.. code-block:: python

    import sknetwork as skn

Graph loading
-------------

A graph is represented by its :term:`adjacency` matrix (square matrix).
When the graph is bipartite, it can be represented by its :term:`biadjacency` matrix (rectangular matrix).

Check our :ref:`tutorial<DataTag>` for various ways of loading a graph
(from a list of edges, a dataframe or a CSV file, for instance).

Graph analysis
--------------

Each algorithm is represented as an object with a ``fit`` method.

Here is an example to cluster the `Karate club graph`_ with the `Louvain algorithm`_:

.. code-block:: python

    from sknetwork.data import karate_club
    from sknetwork.clustering import Louvain

    adjacency = karate_club()
    algo = Louvain()
    algo.fit(adjacency)
    labels = algo.labels_

.. _Karate club graph: https://en.wikipedia.org/wiki/Zachary%27s_karate_club
.. _Louvain algorithm: https://en.wikipedia.org/wiki/Louvain_method
