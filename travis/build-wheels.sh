#!/bin/bash
set -e -x

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
    (cd "$HOME"; "${PYBIN}/py.test" py.test --doctest-modules --cov-report=xml --cov=sknetwork)
done
