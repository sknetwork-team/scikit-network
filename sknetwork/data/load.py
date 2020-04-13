#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on November 15, 2019
@author: Quentin Lutz <qlutz@enst.fr>
This module's code is freely adapted from TorchKGE's torchkge.data.DataLoader.py code.
"""

import pickle
import tarfile
import shutil
from os import environ, makedirs, remove, listdir, rmdir
from os.path import exists, expanduser, join
from typing import Optional
from urllib.error import HTTPError
from urllib.request import urlretrieve

import numpy as np
from scipy import sparse

from sknetwork.data.parse import parse_tsv, parse_labels, parse_header, parse_metadata
from sknetwork.utils import Bunch


def get_data_home(data_home: Optional[str] = None):
    """
    Return a path to a storage folder depending on the dedicated environment variable and user input.

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
    Clear storage folder depending on the dedicated environment variable and user input.

    Parameters
    ----------
    data_home: str
        The folder to be used for dataset storage
    """
    data_home = get_data_home(data_home)
    shutil.rmtree(data_home)


def load_wikilinks(dataset_name: str, data_home: Optional[str] = None,
                   max_depth: int = 1, full_label: bool = True) -> Bunch:
    """Load a dataset from the `WikiLinks database
    <https://graphs.telecom-paristech.fr/Home_page.html#wikilinks-section>`_.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset (all lowcase). Currently, 'wikivitals' and 'wikihumans' are available.
    data_home : str
        The folder to be used for dataset storage.
    max_depth : int
        Maximum depth to use for the labels (if relevant).
    full_label : bool
        If ``True``, return the full name of the label, at maximum depth;
        otherwise, return only the deepest name in the hierarchy.

    Returns
    -------
    graph : :class:`Bunch`

    Example
    -------
    >>> from sknetwork.data import load_wikilinks
    >>> graph = load_wikilinks('wikivitals')
    >>> graph.adjacency.shape
    (10012, 10012)
    """
    if data_home is None:
        data_home = get_data_home()
    data_path = data_home + '/' + dataset_name + '/'
    if not exists(data_path):
        makedirs(data_path, exist_ok=True)
        try:
            urlretrieve("https://graphs.telecom-paristech.fr/npz_datasets/" + dataset_name + '_npz.tar.gz',
                        data_home + '/' + dataset_name + '_npz.tar.gz')
        except HTTPError:
            rmdir(data_home + '/' + dataset_name)
            raise ValueError('Invalid dataset ' + dataset_name + '.'
                             + "\nPossible datasets are 'wikivitals' and 'wikihumans'.")
        with tarfile.open(data_home + '/' + dataset_name + '_npz.tar.gz', 'r:gz') as tar_ref:
            tar_ref.extractall(data_home)
        remove(data_home + '/' + dataset_name + '_npz.tar.gz')

    graph = Bunch()
    files = [file for file in listdir(data_path)]

    if 'adjacency.npz' in files:
        graph.adjacency = sparse.load_npz(data_path + '/adjacency.npz')
    elif 'biadjacency.npz' in files:
        graph.biadjacency = sparse.load_npz(data_path + '/biadjacency.npz')
    if 'feature_names.npy' in files:
        graph.names_col = np.load(data_path + '/feature_names.npy')
    if 'names.npy' in files:
        graph.names = np.load(data_path + '/names.npy')
        if hasattr(graph, 'biadjacency'):
            graph.names_row = graph.names
    if 'target_names.npy' in files:
        target_names = np.load(data_path + '/target_names.npy')
        tags = []
        for tag in target_names:
            parts = tag.strip().split('.')
            if full_label:
                tags.append(".".join(parts[:min(max_depth, len(parts))]))
            else:
                tags.append(parts[:min(max_depth, len(parts))][-1])
        tags = np.array(tags)
        names_label, graph.labels = np.unique(tags, return_inverse=True)
        graph.labels_name = {i: name for i, name in enumerate(names_label)}

    return graph


def load_konect(dataset_name: str, data_home: Optional[str] = None, auto_numpy_bundle: bool = True) -> Bunch:
    """Load a dataset from the `Konect database
    <http://konect.uni-koblenz.de>`_.

    Parameters
    ----------
    dataset_name: str
        The name of the dataset as specified in the download link (e.g. for the Actor movies dataset, the corresponding
        name is ``'actor-movie'``).
    data_home: str
        The folder to be used for dataset storage
    auto_numpy_bundle: bool
        Denotes if the dataset should be stored in its default format (False) or using Numpy files for faster
        subsequent access to the dataset (True).

    Returns
    -------
    graph : :class:`Bunch`
        An object with the following attributes:

             * `adjacency` or `biadjacency`: the adjacency/biadjacency matrix for the dataset
             * `meta`: a dictionary containing the metadata as specified by Konect
             * each attribute specified by Konect (ent.* file)

    Example
    -------
    >>> from sknetwork.data import load_konect
    >>> graph = load_konect('dolphins')
    >>> graph.adjacency.shape
    (62, 62)
    """
    if data_home is None:
        data_home = get_data_home()
    data_path = data_home + '/' + dataset_name + '/'
    if not exists(data_path):
        makedirs(data_path, exist_ok=True)
        try:
            urlretrieve('http://konect.uni-koblenz.de/downloads/tsv/' + dataset_name + '.tar.bz2',
                        data_home + '/' + dataset_name + '.tar.bz2')
            with tarfile.open(data_home + '/' + dataset_name + '.tar.bz2', 'r:bz2') as tar_ref:
                tar_ref.extractall(data_home)
        except (HTTPError, tarfile.ReadError):
            rmdir(data_home + '/' + dataset_name)
            raise ValueError('Invalid dataset ' + dataset_name + '.'
                             + "\nExamples include 'actor-movie' and 'ego-facebook'."
                             + "\n See 'http://konect.uni-koblenz.de' for the full list.")
        finally:
            remove(data_home + '/' + dataset_name + '.tar.bz2')
    elif exists(data_path + '/' + dataset_name + '_bundle'):
        return load_from_numpy_bundle(dataset_name + '_bundle', data_path)

    data = Bunch()
    files = [file for file in listdir(data_path) if dataset_name in file]

    matrix = [file for file in files if 'out.' in file]
    if matrix:
        file = matrix[0]
        directed, bipartite, weighted = parse_header(data_path + file)
        if bipartite:
            graph = parse_tsv(data_path + file, directed=directed, bipartite=bipartite, weighted=weighted)
            data.biadjacency = graph.biadjacency
        else:
            graph = parse_tsv(data_path + file, directed=directed, bipartite=bipartite, weighted=weighted)
            data.adjacency = graph.adjacency

    metadata = [file for file in files if 'meta.' in file]
    if metadata:
        file = metadata[0]
        data.meta = parse_metadata(data_path + file)

    attributes = [file for file in files if 'ent.' + dataset_name in file]
    if attributes:
        for file in attributes:
            attribute_name = file.split('.')[-1]
            data[attribute_name] = parse_labels(data_path + file)

    if auto_numpy_bundle:
        save_to_numpy_bundle(data, dataset_name + '_bundle', data_path)

    return data


def save_to_numpy_bundle(data: Bunch, bundle_name: str, data_home: Optional[str] = None):
    """Save a Bunch to a collection of Numpy and Pickle files for faster subsequent loads.

    Parameters
    ----------
    data: Bunch
        The data to save
    bundle_name: str
        The name to be used for the bundle folder
    data_home: str
        The folder to be used for dataset storage
    """
    data_path = data_home + bundle_name
    makedirs(data_path, exist_ok=True)
    for attribute in data:
        if type(data[attribute]) == sparse.csr_matrix:
            sparse.save_npz(data_path + '/' + attribute, data[attribute])
        elif type(data[attribute]) == np.ndarray:
            np.save(data_path + '/' + attribute, data[attribute])
        elif type(data[attribute]) == Bunch:
            pickle.dump(data[attribute], open(data_path + '/' + attribute + '.p', 'wb'))
        else:
            raise TypeError('Unsupported data attribute type '+str(type(data[attribute])) + '.')


def load_from_numpy_bundle(bundle_name: str, data_home: Optional[str] = None):
    """Load a Bunch from a collection of Numpy and Pickle files (inverse function of ``save_to_numpy_bundle``).

    Parameters
    ----------
    bundle_name: str
        The name used for the bundle folder
    data_home: str
        The folder used for dataset storage

    Returns
    -------
    data: Bunch
        The original data
    """
    data_path = data_home + bundle_name
    if not exists(data_path):
        raise FileNotFoundError('No bundle at ' + data_path)
    else:
        files = listdir(data_path)
        data = Bunch()
        for file in files:
            file_name, file_extension = file.split('.')
            if file_extension == 'npz':
                data[file_name] = sparse.load_npz(data_path + '/' + file)
            elif file_extension == 'npy':
                data[file_name] = np.load(data_path + '/' + file)
            elif file_extension == 'p':
                data[file_name] = pickle.load(open(data_path + '/' + file, 'rb'))
        return data
