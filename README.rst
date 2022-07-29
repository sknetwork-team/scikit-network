.. image:: https://perso.telecom-paristech.fr/bonald/logo_sknetwork.png
    :align: right
    :width: 100px
    :alt: logo sknetwork



.. image:: https://img.shields.io/pypi/v/scikit-network.svg
        :target: https://pypi.python.org/pypi/scikit-network

.. image:: https://github.com/sknetwork-team/scikit-network/actions/workflows/ci_checks.yml/badge.svg
        :target: https://github.com/sknetwork-team/scikit-network/actions/workflows/ci_checks.yml

.. image:: https://readthedocs.org/projects/scikit-network/badge/?version=latest
        :target: https://scikit-network.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://codecov.io/gh/sknetwork-team/scikit-network/branch/master/graph/badge.svg
        :target: https://codecov.io/gh/sknetwork-team/scikit-network

.. image:: https://img.shields.io/pypi/pyversions/scikit-network.svg
        :target: https://pypi.python.org/pypi/scikit-network

Python package for the analysis of large graphs:

* Memory-efficient representation of graphs as sparse matrices in scipy_ format
* Fast algorithms
* Simple API inspired by scikit-learn_

.. _scipy: https://www.scipy.org
.. _scikit-learn: https://scikit-learn.org/

Resources
---------

* Free software: BSD license
* GitHub: https://github.com/sknetwork-team/scikit-network
* Documentation: https://scikit-network.readthedocs.io


Quick Start
-----------

Install scikit-network:

.. code-block:: console

    $ pip install scikit-network

Import scikit-network::

    import sknetwork

See our `tutorials <https://scikit-network.readthedocs.io/en/latest/tutorials/data/index.html>`_;
the notebooks are available `here <https://github.com/sknetwork-team/scikit-network/tree/master/docs/tutorials>`_.

You can also have a look at some `use cases <https://scikit-network.readthedocs.io/en/latest/use_cases/text.html>`_.

Citing
------

If you want to cite scikit-network, please refer to the publication in
the `Journal of Machine Learning Research <https://jmlr.org>`_:

.. code::

    @article{JMLR:v21:20-412,
      author  = {Thomas Bonald and Nathan de Lara and Quentin Lutz and Bertrand Charpentier},
      title   = {Scikit-network: Graph Analysis in Python},
      journal = {Journal of Machine Learning Research},
      year    = {2020},
      volume  = {21},
      number  = {185},
      pages   = {1-6},
      url     = {http://jmlr.org/papers/v21/20-412.html}
    }
