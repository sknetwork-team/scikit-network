#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Setup script."""

import numpy

from setuptools import find_packages, setup, Extension
from sysconfig import get_platform
from distutils.command.build_ext import build_ext
import os
import builtins
from glob import glob

test_requirements = ['pytest', 'nose', 'pluggy>=0.7.1']

# if any problems occur with macOS' clang not knowing the -fopenmp flag, see:
# https://stackoverflow.com/questions/43555410/enable-openmp-support-in-clang-in-mac-os-x-sierra-mojave?rq=1
# https://stackoverflow.com/questions/41292059/compiling-cython-with-openmp-support-on-osx

# handling Mac OSX specifics for C++
# taken from https://github.com/huggingface/neuralcoref/blob/master/setup.py on 09/04/2020 (dd/mm)
COMPILE_OPTIONS = {"other": []}
LINK_OPTIONS = {"other": []}

EXTRA_COMPILE_ARGS = ['-fopenmp']
EXTRA_LINK_ARGS = ['-fopenmp']

# Check whether we're on MacOSX >= 10.7
platform = get_platform()
if platform.startswith("macosx"):
    EXTRA_COMPILE_ARGS = []
    EXTRA_LINK_ARGS = []
    # EXTRA_COMPILE_ARGS = ['-lomp']
    # EXTRA_LINK_ARGS = ['-lomp']
    version = platform.split("-")[1].split(".")
    if int(version[0]) > 10 or (int(version[0]) == 10 and int(version[1]) >= 7):
        COMPILE_OPTIONS["other"].append("-stdlib=libc++")
        LINK_OPTIONS["other"].append("-lc++")

# Windows does not (yet) support OpenMP
if platform.startswith("win"):
    # MSVC specific flags:
    # /EHsc: enable C++ exceptions
    # /O2: optimize
    # /d2FH4-: workaround for MSVC function-level hotpatch flag
    EXTRA_COMPILE_ARGS = ['/EHsc', '/O2', '/d2FH4-']
    EXTRA_LINK_ARGS = []

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

    def finalize_options(self):
        build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        builtins.__NUMPY_SETUP__ = False
        self.include_dirs.append(numpy.get_include())


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

if os.environ.get('WITH_CYTHON_PROFILE') is not None:
    ext_define_macros = [('CYTHON_TRACE_NOGIL', '1')]
    compiler_directives = {'linetrace': True}
else:
    ext_define_macros = []
    compiler_directives = {}

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

        ext_modules += cythonize(Extension(name=mod_name,
                                           sources=[pyx_path],
                                           include_dirs=[numpy.get_include()],
                                           language='c++',
                                           extra_compile_args=EXTRA_COMPILE_ARGS,
                                           extra_link_args=EXTRA_LINK_ARGS,
                                           define_macros=[('NOMINMAX', None)] + ext_define_macros),
                                 annotate=True,
                                 compiler_directives=compiler_directives)
else:
    ext_modules = [Extension(modules[index], [c_paths[index]], include_dirs=[numpy.get_include()], language='c++',
                   define_macros=[('NOMINMAX', None)], extra_compile_args=EXTRA_COMPILE_ARGS, extra_link_args=EXTRA_LINK_ARGS)
                   for index in range(len(modules))]


setup(
    packages=find_packages(),
    test_suite='tests',
    tests_require=test_requirements,
    zip_safe=False,
    ext_modules=ext_modules,
    include_dirs=[numpy.get_include()],
    cmdclass={"build_ext": BuildExtSubclass}
)
