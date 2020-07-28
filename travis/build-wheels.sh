#!/bin/bash
# Script to be used on Linux docker images upon release


set -e -x

# Add/remove Python versions here (desired distribution must be present on the quay.io docker images)
versions="/opt/python/cp36-cp36m/bin /opt/python/cp37-cp37m/bin /opt/python/cp38-cp38/bin"

# Compile wheels
for PYBIN in $versions; do
    "${PYBIN}/pip" install -r /io/requirements_dev.txt
    "${PYBIN}/pip" wheel /io/ -w wheelhouse/
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/*.whl; do
    auditwheel repair "$whl" --plat $PLAT -w /io/wheelhouse/
done

# Install packages and test
for PYBIN in $versions; do
    "${PYBIN}/pip" install scikit-network --no-index -f /io/wheelhouse
    (cd "$HOME"; "${PYBIN}/nosetests" sknetwork --with-doctest)
done
