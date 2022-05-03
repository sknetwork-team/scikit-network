#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on November 15, 2019
@author: Quentin Lutz <qlutz@enst.fr>
"""

import pickle
import shutil
import tarfile
from os import environ, makedirs, remove, listdir, rmdir
from os.path import exists, expanduser, join
from pathlib import Path
from typing import Optional, Union
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

import numpy as np
from scipy import sparse

from sknetwork.data.parse import from_csv, load_labels, load_header, load_metadata
from sknetwork.utils import Bunch
from sknetwork.utils.check import is_square
from sknetwork.utils.verbose import Log

NETSET_URL = 'https://netset.telecom-paris.fr'


def get_data_home(data_home: Optional[Union[str, Path]] = None) -> Path:
    """Return a path to a storage folder depending on the dedicated environment variable and user input.

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
    return Path(data_home)


def clear_data_home(data_home: Optional[Union[str, Path]] = None):
    """Clear storage folder depending on the dedicated environment variable and user input.

    Parameters
    ----------
    data_home: str or :class:`pathlib.Path`
        The folder to be used for dataset storage
    """
    data_home = get_data_home(data_home)
    shutil.rmtree(data_home)


def load_netset(name: Optional[str] = None, data_home: Optional[Union[str, Path]] = None,
                verbose: bool = True) -> Bunch:
    """Load a dataset from the `NetSet database
    <https://netset.telecom-paris.fr/>`_.

    Parameters
    ----------
    name : str
        Name of the dataset (all low-case). Examples include 'openflights', 'cinema' and 'wikivitals'.
    data_home : str or :class:`pathlib.Path`
        Folder to be used for dataset storage.
    verbose : bool
        Enable verbosity.

    Returns
    -------
    dataset : :class:`Bunch`
        Returned dataset.
    """
    dataset = Bunch()
    npz_folder = NETSET_URL + '/datasets_npz/'
    logger = Log(verbose)

    if name is None:
        print("Please specify the dataset (e.g., 'openflights' or 'wikivitals').\n" +
              f"Complete list available here: <{npz_folder}>.")
        return dataset
    else:
        name = name.lower()
    data_home = get_data_home(data_home)
    data_path = data_home / name
    if not data_path.exists():
        makedirs(data_path, exist_ok=True)
        try:
            logger.print('Downloading', name, 'from NetSet...')
            urlretrieve(npz_folder + name + '_npz.tar.gz',
                        data_home / (name + '_npz.tar.gz'))
        except HTTPError:
            rmdir(data_path)
            raise ValueError('Invalid dataset: ' + name + '.'
                             + "\nAvailable datasets include 'openflights' and 'wikivitals'."
                             + f"\nSee <{NETSET_URL}>")
        except ConnectionResetError:  # pragma: no cover
            rmdir(data_path)
            raise RuntimeError("Could not reach Netset.")
        with tarfile.open(data_home / (name + '_npz.tar.gz'), 'r:gz') as tar_ref:
            logger.print('Unpacking archive...')
            tar_ref.extractall(data_home)
        remove(data_home / (name + '_npz.tar.gz'))

    files = [file for file in listdir(data_path)]
    logger.print('Parsing files...')
    for file in files:
        file_components = file.split('.')
        if len(file_components) == 2:
            file_name, file_extension = tuple(file_components)
            if file_extension == 'npz':
                dataset[file_name] = sparse.load_npz(data_path / file)
            elif file_extension == 'npy':
                dataset[file_name] = np.load(data_path / file, allow_pickle=True)
            elif file_extension == 'p':
                with open(data_path / file, 'rb') as f:
                    dataset[file_name] = pickle.load(f)

    logger.print('Done.')
    return dataset


def load_konect(name: str, data_home: Optional[Union[str, Path]] = None, auto_numpy_bundle: bool = True,
                verbose: bool = True) -> Bunch:
    """Load a dataset from the `Konect database
    <http://konect.cc/networks/>`_.

    Parameters
    ----------
    name : str
        Name of the dataset as specified on the Konect website (e.g. for the Zachary Karate club dataset,
        the corresponding name is ``'ucidata-zachary'``).
    data_home : str or :class:`pathlib.Path`
        Folder to be used for dataset storage.
    auto_numpy_bundle : bool
        Whether the dataset should be stored in its default format (False) or using Numpy files for faster
        subsequent access to the dataset (True).
    verbose : bool
        Enable verbosity.

    Returns
    -------
    dataset : :class:`Bunch`
        Object with the following attributes:

             * `adjacency` or `biadjacency`: the adjacency/biadjacency matrix for the dataset
             * `meta`: a dictionary containing the metadata as specified by Konect
             * each attribute specified by Konect (ent.* file)

    Notes
    -----
    An attribute `meta` of the `Bunch` class is used to store information about the dataset if present. In any case,
    `meta` has the attribute `name` which, if not given, is equal to the name of the dataset as passed to this function.

    References
    ----------
    Kunegis, J. (2013, May).
    `Konect: the Koblenz network collection.
    <https://dl.acm.org/doi/abs/10.1145/2487788.2488173>`_
    In Proceedings of the 22nd International Conference on World Wide Web (pp. 1343-1350).
    """
    logger = Log(verbose)
    if name == '':
        raise ValueError("Please specify the dataset. "
                         + "\nExamples include 'actor-movie' and 'ego-facebook'."
                         + "\n See 'http://konect.cc/networks/' for the full list.")
    data_home = get_data_home(data_home)
    data_path = data_home / name
    if not data_path.exists():
        logger.print('Downloading', name, 'from Konect...')
        makedirs(data_path, exist_ok=True)
        try:
            urlretrieve('http://konect.cc/files/download.tsv.' + name + '.tar.bz2',
                        data_home / (name + '.tar.bz2'))
            with tarfile.open(data_home / (name + '.tar.bz2'), 'r:bz2') as tar_ref:
                logger.print('Unpacking archive...')
                tar_ref.extractall(data_home)
        except (HTTPError, tarfile.ReadError):
            rmdir(data_path)
            raise ValueError('Invalid dataset ' + name + '.'
                             + "\nExamples include 'actor-movie' and 'ego-facebook'."
                             + "\n See 'http://konect.cc/networks/' for the full list.")
        except (URLError, ConnectionResetError):  # pragma: no cover
            rmdir(data_path)
            raise RuntimeError("Could not reach Konect.")
        finally:
            if exists(data_home / (name + '.tar.bz2')):
                remove(data_home / (name + '.tar.bz2'))
    elif exists(data_path / (name + '_bundle')):
        logger.print('Loading from local bundle...')
        return load_from_numpy_bundle(name + '_bundle', data_path)

    dataset = Bunch()

    files = [file for file in listdir(data_path) if name in file]
    logger.print('Parsing files...')
    matrix = [file for file in files if 'out.' in file]
    if matrix:
        file = matrix[0]
        directed, bipartite, weighted = load_header(data_path / file)
        dataset = from_csv(data_path / file, directed=directed, bipartite=bipartite, weighted=weighted)

    metadata = [file for file in files if 'meta.' in file]
    if metadata:
        file = metadata[0]
        dataset.meta = load_metadata(data_path / file)

    attributes = [file for file in files if 'ent.' + name in file]
    if attributes:
        for file in attributes:
            attribute_name = file.split('.')[-1]
            dataset[attribute_name] = load_labels(data_path / file)

    if hasattr(dataset, 'meta'):
        if hasattr(dataset.meta, 'name'):
            pass
        else:
            dataset.meta.name = name
    else:
        dataset.meta = Bunch()
        dataset.meta.name = name

    if auto_numpy_bundle:
        save_to_numpy_bundle(dataset, name + '_bundle', data_path)

    return dataset


def save_to_numpy_bundle(data: Bunch, bundle_name: str, data_home: Optional[Union[str, Path]] = None):
    """Save a Bunch in the specified data home to a collection of Numpy and Pickle files for faster subsequent loads.

    Parameters
    ----------
    data: Bunch
        Data to save.
    bundle_name: str
        Name to be used for the bundle folder.
    data_home: str or :class:`pathlib.Path`
        Folder to be used for dataset storage.
    """
    data_home = get_data_home(data_home)
    data_path = data_home / bundle_name
    makedirs(data_path, exist_ok=True)
    for attribute in data:
        if type(data[attribute]) == sparse.csr_matrix:
            sparse.save_npz(data_path / attribute, data[attribute])
        elif type(data[attribute]) == np.ndarray:
            np.save(data_path / attribute, data[attribute])
        elif type(data[attribute]) == Bunch or type(data[attribute]) == str:
            with open(data_path / (attribute + '.p'), 'wb') as file:
                pickle.dump(data[attribute], file)
        else:
            raise TypeError('Unsupported data attribute type '+str(type(data[attribute])) + '.')


def load_from_numpy_bundle(bundle_name: str, data_home: Optional[Union[str, Path]] = None):
    """Load a Bunch from a collection of Numpy and Pickle files (inverse function of ``save_to_numpy_bundle``).

    Parameters
    ----------
    bundle_name: str
        Name of the bundle folder.
    data_home: str or :class:`pathlib.Path`
        Folder used for dataset storage.

    Returns
    -------
    data: Bunch
        Data.
    """
    data_home = get_data_home(data_home)
    data_path = data_home / bundle_name
    if not data_path.exists():
        raise FileNotFoundError('No bundle at ' + str(data_path))
    else:
        files = listdir(data_path)
        data = Bunch()
        for file in files:
            if len(file.split('.')) == 2:
                file_name, file_extension = file.split('.')
                if file_extension == 'npz':
                    data[file_name] = sparse.load_npz(data_path / file)
                elif file_extension == 'npy':
                    data[file_name] = np.load(data_path / file, allow_pickle=True)
                elif file_extension == 'p':
                    with open(data_path / file, 'rb') as f:
                        data[file_name] = pickle.load(f)
        return data


def save(folder: Union[str, Path], data: Union[sparse.csr_matrix, Bunch]):
    """Save a Bunch or a CSR matrix in the current directory to a collection of Numpy and Pickle files for faster
    subsequent loads. Supported attribute types include sparse matrices, NumPy arrays, strings and Bunch.

    Parameters
    ----------
    folder : str or :class:`pathlib.Path`
        Name of the bundle folder.
    data : Union[sparse.csr_matrix, Bunch]
        Data to save.

    Example
    -------
    >>> from sknetwork.data import save
    >>> my_dataset = Bunch()
    >>> my_dataset.adjacency = sparse.csr_matrix(np.random.random((3, 3)) < 0.5)
    >>> my_dataset.names = np.array(['a', 'b', 'c'])
    >>> save('my_dataset', my_dataset)
    >>> 'my_dataset' in listdir('.')
    True
    """
    folder = Path(folder)
    folder = folder.expanduser()
    if folder.exists():
        shutil.rmtree(folder)
    if isinstance(data, sparse.csr_matrix):
        bunch = Bunch()
        if is_square(data):
            bunch.adjacency = data
        else:
            bunch.biadjacency = data
        data = bunch
    if folder.is_absolute():
        save_to_numpy_bundle(data, folder, '/')
    else:
        save_to_numpy_bundle(data, folder, '.')


def load(folder: Union[str, Path]):
    """Load a Bunch from a previously created bundle from the current directory (inverse function of ``save``).

    Parameters
    ----------
    folder: str
        Name of the bundle folder.

    Returns
    -------
    data: Bunch
        Data.

    Example
    -------
    >>> from sknetwork.data import save
    >>> my_dataset = Bunch()
    >>> my_dataset.adjacency = sparse.csr_matrix(np.random.random((3, 3)) < 0.5)
    >>> my_dataset.names = np.array(['a', 'b', 'c'])
    >>> save('my_dataset', my_dataset)
    >>> loaded_graph = load('my_dataset')
    >>> loaded_graph.names[0]
    'a'
    """
    folder = Path(folder)
    if folder.is_absolute():
        return load_from_numpy_bundle(folder, '/')
    else:
        return load_from_numpy_bundle(folder, '.')
