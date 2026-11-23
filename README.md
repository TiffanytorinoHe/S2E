# S2E: Simulator-to-Equation

Source code, fixed inputs and analysis data for **Deriving analytical theory from biophysically detailed neuron simulators reveals principles of dendritic inhibition**.

This repository contains the corrected S2E implementation used for the inhibition study. It covers four detailed neuron models, the transfer–drive analysis, autonomous voltage validation, a reduced-model enumeration benchmark and the L5 inhibitory spike-train control.

## Start with the data and figures

The statistical analysis and figure exports run on CPU; a GPU is needed for detailed inhibitory search and profile collection. Commands below start in this directory.

```bash
conda env create -f environment/analysis.yml
conda activate s2e-analysis
python analysis/pulse_statistics.py
python analysis/train_statistics.py
python analysis/transfer_drive.py
python analysis/iteration_alignment.py
python analysis/verify_release.py
python figures/figure3.py
python figures/main_panels.py
python figures/supplementary.py
```

Outputs go to `outputs/`. The verification command checks 800 fixed inputs, completed solutions, source identities, 800 initial profiles, the 298-case enumeration and its CPU certificates. The statistical scripts compare recomputed tables against the frozen publication data. Figures are exported as PNG, PDF and SVG with editable text. `figures/main_panels.py` reconstructs numerical panels; it does not reproduce the illustrator's conceptual artwork or composite layout. The six author-supplied main figures accompany the manuscript separately.

## Run the scientific model

See [environment/README.md](environment/README.md) for Linux/CUDA installation and [experiments/README.md](experiments/README.md) for individual runs and continuation. Use the exact case IDs in `data/cohorts.json`, rather than generating new random inputs. Each process owns one GPU and one realization. Simulation time step, original morphology, channel parameters, model-loading temperature, initialization and inputs are retained by the model-specific runtime. The S2E inhibition cohort is separate from the morphology-growth study; no growth experiment or optimizer atlas is included here.

## Contents

| Directory | Purpose |
|---|---|
| `models/` | Frozen detailed model, tangent and adjoint implementations; MOD/HOC/morphology assets |
| `src/voltage/` | Autonomous voltage, channel and synaptic update equations |
| `src/profiles/` | Resident CUDA transfer/drive collection |
| `src/bridges/` | NEURON state/matrix extraction extensions |
| `src/neurong/` | GPU engine dependency for the SIMPLE enumeration benchmark |
| `experiments/` | Portable single-case search, profile, voltage and benchmark entry points |
| `analysis/` | Cohort, statistics, identity and validation checks |
| `figures/` | Numerical main panels and complete supplementary figures |
| `data/` | Frozen inputs, solutions, compact profiles, plot data and validation records |
| `environment/` | Environment specifications, pinned NEURON revision and build recipes |

See [data/README.md](data/README.md) for cohort definitions and [figures/README.md](figures/README.md) for the paper-to-code map.

## Validation

The retained scientific source files match their recorded SHA256 manifests. The package's CPU analysis and plotting commands were executed, including recomputation of Figure 5, Supplementary Table 6 and twelve Figure 6 data tables. These commands regenerate validation outputs under `outputs/verification/` and `outputs/analysis/`; generated outputs are excluded from Git.

The portable Linux build and simulation entry points were assembled from the validated production implementations and checked statically. A complete clean Linux/CUDA rebuild of this newly organized package has not yet been executed. Installation time and fresh-platform GPU runtimes therefore remain to be measured. Historical scientific validation is included in `data/validation/`; it is not a clean-install test of the packaging layer. A fresh reproduction must pass the included CPU/GPU/finite-difference gate before proceeding.

## Release and citation

The simulation and analysis code, figure-generation scripts, fixed simulation inputs and figure source data are publicly available in [this repository](https://github.com/TiffanytorinoHe/S2E). The manuscript uses [commit `1aac458`](https://github.com/TiffanytorinoHe/S2E/tree/1aac458b40972d9ec158e2c72e8e310070673795), tagged `v0.1.0-review`.

`CITATION.cff` records the manuscript authors and title. See [THIRD_PARTY.md](THIRD_PARTY.md) for model and dependency attributions and information on reuse.

The packaging follows the installation, example, validation and data-documentation items in the [Nature code submission guidance](https://media.nature.com/full/nature-cms/documents/GuidelinesCodePublication.pdf) and [computational reproducibility guidance](https://www.nature.com/documents/Computational_tools_reporting_guidelines.pdf). The target journal's submission requirements should be checked when selected.
