#!/usr/bin/env bash
S2E_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PATH="$S2E_ROOT/environment/build/neuron-install/bin:$PATH"
export PYTHONPATH="$S2E_ROOT/environment/build/neuron-install/lib/python:$S2E_ROOT/environment/build/bridges:$S2E_ROOT/environment/build/neurong-install/python:$S2E_ROOT/environment/build/neurong-install/lib/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$S2E_ROOT/environment/build/neuron-install/lib:$S2E_ROOT/environment/build/neurong-install/lib${CONDA_PREFIX:+:$CONDA_PREFIX/lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
