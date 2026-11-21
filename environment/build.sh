#!/usr/bin/env bash
set -euo pipefail
S2E_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
S2E_ENV="$S2E_ROOT/environment"
S2E_COMMIT=e0290762790622291c4867dee25a7c13e0c2d326
S2E_SRC="$S2E_ENV/build/neuron-source"
S2E_BUILD="$S2E_ENV/build/neuron-build"
S2E_PREFIX="$S2E_ENV/build/neuron-install"
mkdir -p "$S2E_ENV/build"
if [ ! -d "$S2E_SRC/.git" ]; then
 git clone https://github.com/neuronsimulator/nrn.git "$S2E_SRC"
 git -C "$S2E_SRC" checkout "$S2E_COMMIT"
 git -C "$S2E_SRC" submodule update --init --recursive
fi
test "$(git -C "$S2E_SRC" rev-parse HEAD)" = "$S2E_COMMIT"
# The supplied source changes only the fmatrix node-index diagnostic selector.
cp "$S2E_ENV/fadvance.cpp" "$S2E_SRC/src/nrnoc/fadvance.cpp"
export SETUPTOOLS_SCM_PRETEND_VERSION=9.0.dev1376+ge029076
cmake -S "$S2E_SRC" -B "$S2E_BUILD" -G Ninja \
 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$S2E_PREFIX" \
 -DPYTHON_EXECUTABLE="$(command -v python)" -DNRN_ENABLE_INTERVIEWS=OFF \
 -DNRN_ENABLE_MPI=OFF -DNRN_ENABLE_RX3D=OFF -DNRN_ENABLE_CORENEURON=OFF \
 -DNRN_ENABLE_TESTS=OFF -DNRN_ENABLE_BACKTRACE=OFF
cmake --build "$S2E_BUILD" --parallel 4
cmake --install "$S2E_BUILD"
cmake -S "$S2E_ENV" -B "$S2E_ENV/build/bridges" -G Ninja \
 -DNRN_SOURCE_DIR="$S2E_SRC" -DNRN_BUILD_DIR="$S2E_BUILD" -DNRN_INSTALL_DIR="$S2E_PREFIX" \
 -DPython_EXECUTABLE="$(command -v python)" -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build "$S2E_ENV/build/bridges" --parallel 4
source "$S2E_ENV/activate.sh"
for S2E_MODEL in l5pc ca1 l23 msn_sync msn_nmda_hf msn_ampa; do
 (
  cd "$S2E_ROOT/models/$S2E_MODEL"
  nrnivmodl mod
  nvcc -O3 -std=c++17 -arch="${S2E_CUDA_ARCH:-sm_120}" -Xcompiler -fPIC --shared gpu.cu -o libtangent.so
  nvcc -O3 -std=c++17 -arch="${S2E_CUDA_ARCH:-sm_120}" -Xcompiler -fPIC --shared adjoint.cu -o libadjoint.so
 )
done
(
 cd "$S2E_ROOT/src/profiles"
 nvcc -O3 -std=c++17 -arch="${S2E_CUDA_ARCH:-sm_120}" -Xcompiler -fPIC --shared resident.cu -o libprofiles.so
)
python -c 'import neuron,data_c2py,data_c2py_v2; print(neuron.h.nrnversion())'
