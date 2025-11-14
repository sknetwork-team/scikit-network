#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in November 2019
@author: Quentin Lutz <qlutz@enst.fr>
"""

import pickle
import shutil
import tarfile
from os import environ, makedirs, remove, listdir
from os.path import abspath, commonprefix, exists, expanduser, isfile, join
from pathlib import Path
from typing import Optional, Union
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

import numpy as np
from scipy import sparse

from sknetwork.data.parse import from_csv, load_labels, load_header, load_metadata
from sknetwork.data.base import Dataset
from sknetwork.utils.check import is_square
from sknetwork.log import Log

NETSET_URL = 'https://netset.telecom-paris.fr'


def is_within_directory(directory, target):
    """Utility function."""
    abs_directory = abspath(directory)
    abs_target = abspath(target)
    prefix = commonprefix([abs_directory, abs_target])
    return prefix == abs_directory


def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
    """Safe extraction."""
    for member in tar.getmembers():
        member_path = join(path, member.name)
        if not is_within_directory(path, member_path):
            raise Exception("Attempted path traversal in tar file.")
    tar.extractall(path, members, numeric_owner=numeric_owner)


def get_data_home(data_home: Optional[Union[str, Path]] = None) -> Path:
    """Return a path to a storage folder depending on the dedicated environment variable and user input.

    Parameters
    ----------
    data_home: str
        The folder to be used for dataset storage
    """
    if data_home is None:
        data_home = environ.get('SCIKIT_NETWORK_DATA', join('~', 'scikit_network_data'))
    data_home = expanduser(data_home)
    if not exists(data_home):
        makedirs(data_home)
    return Path(data_home)


def clear_data_home(data_home: Optional[Union[str, Path]] = None):
    """Clear storage folder.

    Parameters
    ----------
    data_home: str or :class:`pathlib.Path`
        The folder to be used for dataset storage.
    """
    data_home = get_data_home(data_home)
    shutil.rmtree(data_home)


def clean_data_home(data_home: Optional[Union[str, Path]] = None):
    """Clean storage folder so that it contains folders only.

    Parameters
    ----------
    data_home: str or :class:`pathlib.Path`
        The folder to be used for dataset storage
    """
    data_home = get_data_home(data_home)
    for file in listdir(data_home):
        if isfile(join(data_home, file)):
            remove(join(data_home, file))


def load_netset(name: Optional[str] = None, data_home: Optional[Union[str, Path]] = None,
                verbose: bool = True) -> Optional[Dataset]:
    """Load a dataset from the `NetSet collection
    <https://netset.telecom-paris.fr/>`_.

    Parameters
    ----------
    name : str
        Name of the dataset (all low-case). Examples include 'openflights', 'cinema' and 'wikivitals'.
    data_home : str or :class:`pathlib.Path`
        Folder to be used for dataset storage.
        This folder must be empty or contain other folders (datasets); files will be removed.
    verbose : bool
        Enable verbosity.

    Returns
    -------
    dataset : :class:`Dataset`
        Returned dataset.
    """
    dataset = Dataset()
    dataset_folder = NETSET_URL + '/datasets/'
    folder_npz = NETSET_URL + '/datasets_npz/'

    logger = Log(verbose)

    if name is None:
        print("Please specify the dataset (e.g., 'wikivitals').\n" +
              f"Complete list available here: <{dataset_folder}>.")
        return None
    else:
        name = name.lower()
    data_home = get_data_home(data_home)
    data_netset = data_home / 'netset'
    if not data_netset.exists():
        clean_data_home(data_home)
        makedirs(data_netset)

    # remove previous dataset if not in the netset folder
    direct_path = data_home / name
    if direct_path.exists():
        shutil.rmtree(direct_path)

    data_path = data_netset / name
    if not data_path.exists():
        name_npz = name + '_npz.tar.gz'
        try:
            logger.print_log('Downloading', name, 'from NetSet...')
            urlretrieve(folder_npz + name_npz, data_netset / name_npz)
        except HTTPError:
            raise ValueError('Invalid dataset: ' + name + '.'
                             + "\nAvailable datasets include 'openflights' and 'wikivitals'."
                             + f"\nSee <{NETSET_URL}>")
        except ConnectionResetError:  # pragma: no cover
            raise RuntimeError("Could not reach Netset.")
        with tarfile.open(data_netset / name_npz, 'r:gz') as tar_ref:
            logger.print_log('Unpacking archive...')
            safe_extract(tar_ref, data_path)

    files = [file for file in listdir(data_path)]
    logger.print_log('Parsing files...')
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

    clean_data_home(data_netset)
    logger.print_log('Done.')
    return dataset


def save_to_numpy_bundle(data: Dataset, bundle_name: str, data_home: Optional[Union[str, Path]] = None):
    """Save a dataset in the specified data home to a collection of Numpy and Pickle files for faster subsequent loads.

    Parameters
    ----------
    data: Dataset
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
        else:
            with open(data_path / (attribute + '.p'), 'wb') as file:
                pickle.dump(data[attribute], file)


def load_from_numpy_bundle(bundle_name: str, data_home: Optional[Union[str, Path]] = None):
    """Load a dataset from a collection of Numpy and Pickle files (inverse function of ``save_to_numpy_bundle``).

    Parameters
    ----------
    bundle_name: str
        Name of the bundle folder.
    data_home: str or :class:`pathlib.Path`
        Folder used for dataset storage.

    Returns
    -------
    data: Dataset
        Data.
    """
    data_home = get_data_home(data_home)
    data_path = data_home / bundle_name
    if not data_path.exists():
        raise FileNotFoundError('No bundle at ' + str(data_path))
    else:
        files = listdir(data_path)
        data = Dataset()
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


def save(folder: Union[str, Path], data: Union[sparse.csr_matrix, Dataset]):
    """Save a dataset or a CSR matrix in the current directory to a collection of Numpy and Pickle files for faster
    subsequent loads. Supported attribute types include sparse matrices, NumPy arrays, strings and objects Dataset.

    Parameters
    ----------
    folder : str or :class:`pathlib.Path`
        Name of the bundle folder.
    data : Union[sparse.csr_matrix, Dataset]
        Data to save.

    Example
    -------
    >>> from sknetwork.data import save
    >>> dataset = Dataset()
    >>> dataset.adjacency = sparse.csr_matrix(np.random.random((3, 3)) < 0.5)
    >>> dataset.names = np.array(['a', 'b', 'c'])
    >>> save('dataset', dataset)
    >>> 'dataset' in listdir('.')
    True
    """
    folder = Path(folder)
    folder = folder.expanduser()
    if folder.exists():
        shutil.rmtree(folder)
    if isinstance(data, sparse.csr_matrix):
        dataset = Dataset()
        if is_square(data):
            dataset.adjacency = data
        else:
            dataset.biadjacency = data
        data = dataset
    if folder.is_absolute():
        save_to_numpy_bundle(data, folder, '/')
    else:
        save_to_numpy_bundle(data, folder, '.')


def load(folder: Union[str, Path]):
    """Load a dataset from a previously created bundle from the current directory (inverse function of ``save``).

    Parameters
    ----------
    folder: str
        Name of the bundle folder.

    Returns
    -------
    data: Dataset
        Data.

    Example
    -------
    >>> from sknetwork.data import save
    >>> dataset = Dataset()
    >>> dataset.adjacency = sparse.csr_matrix(np.random.random((3, 3)) < 0.5)
    >>> dataset.names = np.array(['a', 'b', 'c'])
    >>> save('dataset', dataset)
    >>> dataset = load('dataset')
    >>> print(dataset.names)
    ['a' 'b' 'c']
    """
    folder = Path(folder)
    if folder.is_absolute():
        return load_from_numpy_bundle(folder, '/')
    else:
        return load_from_numpy_bundle(folder, '.')
