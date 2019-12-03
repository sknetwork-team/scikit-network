#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on November 15, 2019
@author: Quentin Lutz <qlutz@enst.fr>
This module's code is freely adapted from TorchKGE's torchkge.data.DataLoader.py code.
"""

import shutil
import zipfile
from os import environ, makedirs, remove
from os.path import exists, expanduser, join
from typing import Optional
from urllib.request import urlretrieve

import numpy as np

from sknetwork.data.parsing import parse_tsv, parse_labels, parse_hierarchical_labels
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


def get_vital_wikipedia(data_home: Optional[str] = None):
    """
    Downloads the Vital Wikipedia dataset if needed.

    Parameters
    ----------
    data_home: str
        The folder to be used for dataset storage
    """
    if data_home is None:
        data_home = get_data_home()
    data_path = data_home + '/vital_wikipedia'
    if not exists(data_path):
        makedirs(data_path, exist_ok=True)
        urlretrieve("https://graphs.telecom-paristech.fr/datasets/vital_wikipedia.zip",
                    data_home + '/vital_wikipedia.zip')
        with zipfile.ZipFile(data_home + '/vital_wikipedia.zip', 'r') as zip_ref:
            zip_ref.extractall(data_home)
        remove(data_home + '/vital_wikipedia.zip')
    return data_path


def load_vital_wikipedia_links(data_home: Optional[str] = None, max_depth: int = 1, full_path: bool = True):
    """
    Loads the hyperlinks of the `Vital Wikipedia
    <https://graphs.telecom-paristech.fr/Home_page.html#vitalwiki-section>`_ dataset.

    Parameters
    ----------
    data_home: str
        The folder to be used for dataset storage
    max_depth: int
        Denotes the maximum depth to use for the categories
    full_path: bool
        Denotes if only the deepest label possible should be returned or if all super categories should
        be considered (default)

    Returns
    -------
    data: 'Bunch'
        An object with the following attributes:
         * `adjacency`: the adjacency matrix of the hyperlink graph in CSR format
         * `names`: the titles of the articles
         * `target_names`: the categories of the articles as specified with `max_depth` and `full_path`
         * `target`: the index for `target_names`
    """
    data_path = get_vital_wikipedia(data_home)

    data = Bunch()
    data.adjacency = parse_tsv(data_path + '/en-internal-links.txt', directed=True, reindex=False)
    data.names = parse_labels(data_path + '/en-articles.txt')
    data.target_names = parse_hierarchical_labels(data_path + '/en-categories.txt', max_depth, full_path=full_path)
    _, data.target = np.unique(data.target_names, return_inverse=True)

    return data


def load_vital_wikipedia_text(data_home: Optional[str] = None, max_depth: int = 1, full_path: bool = True):
    """
    Loads the stems of the `Vital Wikipedia
    <https://graphs.telecom-paristech.fr/Home_page.html#vitalwiki-section>`_ dataset.

    Parameters
    ----------
    data_home: str
        The folder to be used for dataset storage
    max_depth: int
        Denotes the maximum depth to use for the categories
    full_path: bool
        Denotes if only the deepest label possible should be returned or if all super categories should
        be considered (default)

    Returns
    -------
    data: 'Bunch'
        An object with the following attributes:
         * `biadjacency`: the biadjacency matrix of the stems in CSR format
         * `feature_names`: the array of the stems
         * `names`: the titles of the articles
         * `target_names`: the categories of the articles as specified with `max_depth` and `full_path`
         * `target`: the index for `target_names`
    """
    data_path = get_vital_wikipedia(data_home)

    data = Bunch()
    data.biadjacency = parse_tsv(data_path + '/en-articles-stems.txt', bipartite=True, reindex=False)
    data.feature_names = parse_labels(data_path + '/en-stems.txt')
    data.names = parse_labels(data_path + '/en-articles.txt')
    data.target_names = parse_hierarchical_labels(data_path + '/en-categories.txt', max_depth, full_path=full_path)
    _, data.target = np.unique(data.target_names, return_inverse=True)

    return data
