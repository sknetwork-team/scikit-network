#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""


from setuptools import find_packages, dist
import distutils.util
from distutils.core import setup, Extension
from distutils.command.build_ext import build_ext
import os
from glob import glob

dist.Distribution().fetch_build_eggs(['Cython', 'numpy==1.18.3'])

import numpy

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['numpy>=1.18.3', 'scipy>=1.4.1']

setup_requirements = ['pytest-runner']

test_requirements = ['pytest', 'nose', 'pluggy>=0.7.1']

# if any problems occur with macOS' clang not knowing the -fopenmp flag, see:
# https://stackoverflow.com/questions/43555410/enable-openmp-support-in-clang-in-mac-os-x-sierra-mojave?rq=1
# https://stackoverflow.com/questions/41292059/compiling-cython-with-openmp-support-on-osx

# handling Mac OSX specifics for C++
# taken from https://github.com/huggingface/neuralcoref/blob/master/setup.py on 09/04/2020 (dd/mm)
COMPILE_OPTIONS = {"other": []}
LINK_OPTIONS = {"other": []}

OPENMP_COMPILE_FLAG = '-fopenmp'
OPENMP_LINK_FLAG = '-fopenmp'

# Check whether we're on OSX >= 10.10
name = distutils.util.get_platform()
if name.startswith("macosx-10"):
    OPENMP_COMPILE_FLAG = '-lomp'
    OPENMP_LINK_FLAG = '-lomp'
    minor_version = int(name.split("-")[1].split(".")[1])
    if minor_version >= 7:
        COMPILE_OPTIONS["other"].append("-stdlib=libc++")
        LINK_OPTIONS["other"].append("-lc++")
        # g++ (used by unix compiler on mac) links to libstdc++ as a default lib.
        # See: https://stackoverflow.com/questions/1653047/avoid-linking-to-libstdc
        LINK_OPTIONS["other"].append("-nodefaultlibs")
# Windows does not (yet) support OpenMP
if name.startswith("win"):
    OPENMP_COMPILE_FLAG = ''
    OPENMP_LINK_FLAG = ''


class BuildExtSubclass(build_ext):
    def build_options(self):
        for e in self.extensions:
            e.extra_compile_args += COMPILE_OPTIONS.get(
                self.compiler.compiler_type, COMPILE_OPTIONS["other"]
            )
        for e in self.extensions:
            e.extra_link_args += LINK_OPTIONS.get(
                self.compiler.compiler_type, LINK_OPTIONS["other"]
            )

    def build_extensions(self):
        self.build_options()
        build_ext.build_extensions(self)


# Cython generation/C++ compilation
pyx_paths = glob("./sknetwork/**/*.pyx")
c_paths = ['.' + filename.split('.')[1] + '.cpp' for filename in pyx_paths]
modules = [filename.split('.')[1][1:].replace('/', '.').replace('\\', '.') for filename in pyx_paths]

if os.environ.get('SKNETWORK_DISABLE_CYTHONIZE') is None:
    try:
        import Cython
        HAVE_CYTHON = True
    except ImportError:
        HAVE_CYTHON = False
else:
    HAVE_CYTHON = False


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

        ext_modules += cythonize(Extension(name=mod_name, sources=[pyx_path], include_dirs=[numpy.get_include()],
                                           extra_compile_args=[OPENMP_COMPILE_FLAG],
                                           extra_link_args=[OPENMP_LINK_FLAG]), annotate=True)
else:
    ext_modules = [Extension(modules[index], [c_paths[index]], include_dirs=[numpy.get_include()])
                   for index in range(len(modules))]


setup(
    author="Scikit-network team",
    author_email='bonald@enst.fr',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8'
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
    long_description_content_type='text/x-rst',
    include_package_data=True,
    keywords='sknetwork',
    name='scikit-network',
    packages=find_packages(),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/sknetwork-team/scikit-network',
    version='0.22.0',
    zip_safe=False,
    ext_modules=ext_modules,
    include_dirs=[numpy.get_include()],
    cmdclass={"build_ext": BuildExtSubclass}
)

