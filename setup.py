#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""


from setuptools import find_packages
from distutils.core import setup, Extension
import os

import numpy

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=6.0', 'numpy', 'scipy']

setup_requirements = ['pytest-runner']

test_requirements = ['pytest', 'nose', 'pluggy>=0.7.1']


pyx_paths = ["sknetwork/utils/toto.pyx", "sknetwork/clustering/louvain_core.pyx"]
c_paths = ["sknetwork/utils/toto.cpp", "sknetwork/clustering/louvain_core.cpp"]
modules = ['sknetwork.utils.toto', 'sknetwork.clustering.louvain_core']

"""
try:
    import Cython
    HAVE_CYTHON = True
except ImportError:
    HAVE_CYTHON = False
"""

from Cython.Build import cythonize

ext_modules = cythonize(Extension(name='*', sources='*.pyx', extra_compile_args=['-O3']),
                        language='c++')

"""
if HAVE_CYTHON:
    from Cython.Build import cythonize

    ext_modules = []
    for couple_index in range(len(pyx_paths)):
        pyx_path = pyx_paths[couple_index]
        c_path = c_paths[couple_index]
        mod_name = modules[couple_index]
        if os.path.exists(c_path):
            # Remove C file to force Cython recompile.
            os.remove(c_path)

        ext_modules += cythonize(Extension(name=mod_name, sources=[pyx_path], extra_compile_args=['-O3']),
                                 language='c++')
else:
    ext_modules = [Extension(
        modules[index],
        [c_paths[index]],
        extra_compile_args=['-O3'],
        language='c++'
    ) for index in range(len(modules))]
"""

setup(
    author="Scikit-network team",
    author_email='bonald@enst.fr',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
    ],
    description="Graph algorithms",
    entry_points={
        'console_scripts': [
            'sknetwork=sknetwork.cli:main',
        ],
    },
    install_requires=requirements,
    license="BSD license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='sknetwork',
    name='scikit-network',
    packages=find_packages(),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/sknetwork-team/scikit-network',
    version='0.12.1',
    zip_safe=False,
    ext_modules=ext_modules,
    include_dirs=[numpy.get_include()],
    extra_compile_args=["-std=c++11"],
)
