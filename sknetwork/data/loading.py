#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on November 15, 2019
@author: Quentin Lutz <qlutz@enst.fr>
This module's code is freely adapted from TorchKGE's torchkge.data.DataLoader.py code.
"""

import tarfile
import shutil
import zipfile
from os import environ, makedirs, remove, listdir
from os.path import exists, expanduser, join
from typing import Optional
from urllib.error import HTTPError
from urllib.request import urlretrieve

import numpy as np

from sknetwork.basics import is_bipartite
from sknetwork.data.parsing import parse_tsv, parse_labels, parse_hierarchical_labels, parse_header, parse_metadata
from sknetwork.utils import Bunch


def get_data_home(data_home: Optional[str] = None):
    """
    Returns a path to a storage folder depending on the dedicated environment variable and user input.

    Parameters
    ----------
    data_home: str
        The folder to be used for dataset storage
    """
    if data_home is None:
        data_home = environ.get('SCIKIT_NETWORK_DATA',
                                join('~', 'scikit_network_data'))
    data_home = expanduser(data_home)
    if not exists(data_home):
        makedirs(data_home)
    return data_home


def clear_data_home(data_home: Optional[str] = None):
    """
    Clears a storage folder depending on the dedicated environment variable and user input.

    Parameters
    ----------
    data_home: str
        The folder to be used for dataset storage
    """
    data_home = get_data_home(data_home)
    shutil.rmtree(data_home)


def load_wikilinks_dataset(dataset_name: str, data_home: Optional[str] = None,
                           max_depth: int = 1, full_path: bool = True):
    """
    Loads a dataset from the `WikiLinks database
    <https://graphs.telecom-paristech.fr/Home_page.html#wikilinks-section>`_.

    Parameters
    ----------
    dataset_name: str
        The name of the dataset (no capital letters, all spaces replaced with underscores)
    data_home: str
        The folder to be used for dataset storage
    max_depth: int
        Denotes the maximum depth to use for the categories (if relevant)
    full_path: bool
        Denotes if only the deepest label possible should be returned or if all super categories should
        be considered (if relevant)

    Returns
    -------
    data: :class:`Bunch`
        An object with some of the following attributes (depending on the dataset):

         * `adjacency`: the adjacency matrix of the graph in CSR format
         * `biadjacency`: the biadjacency matrix of the graph in CSR format
         * `feature_names`: the array of the names for the features
         * `names`: the titles of the articles
         * `target_names`: the categories of the articles as specified with `max_depth` and `full_path`
         * `target`: the index for `target_names`

    """
    if data_home is None:
        data_home = get_data_home()
    data_path = data_home + '/' + dataset_name + '/'
    if not exists(data_path):
        makedirs(data_path, exist_ok=True)
        try:
            urlretrieve("https://graphs.telecom-paristech.fr/datasets/" + dataset_name + '.tar.gz',
                        data_home + '/' + dataset_name + '.tar.gz')
        except HTTPError:
            raise ValueError('Invalid dataset ' + dataset_name)
        with tarfile.open(data_home + '/' + dataset_name + '.tar.gz', 'r:gz') as tar_ref:
            tar_ref.extractall(data_home)
        remove(data_home + '/' + dataset_name + '.tar.gz')

    data = Bunch()
    files = [file for file in listdir(data_path)]

    if 'adjacency.txt' in files:
        data.adjacency = parse_tsv(data_path + '/adjacency.txt', directed=True, reindex=False)
    if 'biadjacency.txt' in files:
        data.biadjacency = parse_tsv(data_path + '/biadjacency.txt', bipartite=True, reindex=False)
    if 'names.txt' in files:
        data.names = parse_labels(data_path + '/names.txt')
    if 'feature_names.txt' in files:
        data.feature_names = parse_labels(data_path + '/feature_names.txt')
    if 'categories.txt' in files:
        data.target_names = parse_hierarchical_labels(data_path + '/categories.txt', max_depth, full_path=full_path)
        _, data.target = np.unique(data.target_names, return_inverse=True)

    return data


def load_konect_dataset(dataset_name: str, data_home: Optional[str] = None):
    """
    Loads a dataset from the `Konect database
    <http://konect.uni-koblenz.de>`_.

    Parameters
    ----------
    dataset_name: str
        The name of the dataset as specified in the download link (e.g. for the Actor movies dataset, the corresponding
        name is ``'actor-movie'``).
    data_home: str
        The folder to be used for dataset storage

    Returns
    -------
    data: :class:`Bunch`
        An object with the following attributes:

         * `adjacency` or `biadjacency`: the adjacency/biadjacency matrix for the dataset
         * `meta`: a dictionary containing the metadata as specified by Konect
         * any attribute described in an ent.* file

    """
    if data_home is None:
        data_home = get_data_home()
    data_path = data_home + '/' + dataset_name + '/'
    if not exists(data_path):
        makedirs(data_path, exist_ok=True)
        try:
            urlretrieve('http://konect.uni-koblenz.de/downloads/tsv/' + dataset_name + '.tar.bz2',
                        data_home + '/' + dataset_name + '.tar.bz2')
        except HTTPError:
            raise ValueError('Invalid dataset ' + dataset_name)
        with tarfile.open(data_home + '/' + dataset_name + '.tar.bz2', 'r:bz2') as tar_ref:
            tar_ref.extractall(data_home)
        remove(data_home + '/' + dataset_name + '.tar.bz2')

    data = Bunch()
    files = [file for file in listdir(data_path) if dataset_name in file]

    matrix = [file for file in files if 'out.' in file]
    if matrix:
        file = matrix[0]
        directed, bipartite, weighted = parse_header(data_path + file)
        if bipartite:
            data.biadjacency = parse_tsv(data_path + file, directed=directed, bipartite=bipartite, weighted=weighted)[0]
        else:
            data.adjacency = parse_tsv(data_path + file, directed=directed, bipartite=bipartite, weighted=weighted)[0]

    metadata = [file for file in files if 'meta.' in file]
    if metadata:
        file = metadata[0]
        data.meta = parse_metadata(data_path + file)

    attributes = [file for file in files if 'ent.' + dataset_name in file]
    if attributes:
        for file in attributes:
            attribute_name = file.split('.')[-1]
            data[attribute_name] = parse_labels(data_path + file)

    return data
