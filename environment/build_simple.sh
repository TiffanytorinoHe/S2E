#!/usr/bin/env bash
set -euo pipefail
S2E_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$S2E_ROOT/environment/activate.sh"
cmake -S "$S2E_ROOT/src/neurong" -B "$S2E_ROOT/environment/build/neurong" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$S2E_ROOT/environment/build/neurong-install" -DNEURONG_CUDA_ARCHITECTURES="${S2E_NEURONG_ARCH:-120}" -DENABLE_STRICT_CUDA_MATH=OFF -DPython_EXECUTABLE="$(command -v python)"
cmake --build "$S2E_ROOT/environment/build/neurong" --parallel 4
cmake --install "$S2E_ROOT/environment/build/neurong"
cd "$S2E_ROOT/experiments/simple"
nrnivmodl mod
