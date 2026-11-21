# Environment and build

## CPU analysis

`analysis.yml` pins the environment actually used for the package checks: Python 3.9.23, NumPy 1.23.5, SciPy 1.13.1, pandas 2.3.3 and Matplotlib 3.9.4, on Windows. No NEURON or GPU is needed to recompute statistics and figures from retained results. Arial was used for figure exports; install Arial where licensed or explicitly select a substitute font and inspect the export. Output metrics are independent of the font.

## Scientific simulations

Use Linux x86-64 with an NVIDIA GPU and a compatible NVIDIA driver/CUDA toolkit, a C/C++ compiler, Git and Conda. The production GPU target was RTX 5090 (compute capability 12.0); compiling that target requires CUDA 12.8 or later. CUDA must be available through `nvcc`. The `simulation.yml` recipe supplies Python 3.12, numerical libraries, PyTorch, CMake/Ninja, pybind11/nanobind, HDF5 and parser dependencies. It is a reconstruction recipe, not a newly verified clean-machine environment lock.

```bash
conda env create -f environment/simulation.yml
conda activate s2e-simulation
bash environment/build.sh
source environment/activate.sh
python experiments/solve.py --group l5pc_ampa --case 0 --gate-only --output outputs/demo/l5pc_ampa_000
```

`build.sh` fetches NEURON revision `e0290762790622291c4867dee25a7c13e0c2d326`, restores its submodules, installs the supplied fmatrix diagnostic source, builds NEURON and both state-extraction extensions, compiles the six model mechanism libraries, and compiles the tangent/adjoint/profile CUDA libraries. `neuron_fmatrix.patch`, `fadvance.cpp` and `neuron_submodules.json` preserve the production build details. Internet access is needed only for the upstream source/environment installation, not for the supplied scientific inputs.

For another supported GPU, select its real architecture explicitly before building, for example `export S2E_CUDA_ARCH=sm_86` for RTX 3090. Re-run the scientific gate on that platform; cross-device bitwise identity is not assumed. `build.sh` does not change numerical model parameters. Thread counts are capped at one for CPU math libraries.

The SIMPLE enumeration additionally needs the supplied NEURONG engine:

```bash
bash environment/build_simple.sh
source environment/activate.sh
```

Set `S2E_NEURONG_ARCH=86` for RTX 3090, or retain the default `120` for RTX 5090. The NEURONG strict-CUDA-math setting follows the recorded production benchmark. CPU verification uses the pinned NEURON build with HH tables disabled.

Do not use a stock NEURON wheel with these state-extraction extensions: they depend on the pinned internal data layout and fmatrix selector. Never load different model mechanism libraries in one Python process. Launch each case in a fresh process. The wrappers record actual temperature, initialization, timestep, per-section segmentation and input/source hashes.

The cleaned build has not yet been executed on a fresh Linux/CUDA machine. A release tester should record compiler, CUDA/driver, GPU, installation duration, gate result and end-to-end demo duration. These measurements must be observed rather than inferred from historical cluster timings.
