# S2E: Simulator-to-Equation

Source code, fixed inputs and analysis data for **Deriving analytical theory from biophysically detailed neuron simulators reveals principles of dendritic inhibition**.

This review package contains the corrected S2E implementation used for the inhibition study. It covers four detailed neuron models, the transfer–drive analysis, autonomous voltage validation, a reduced-model enumeration benchmark and the L5 inhibitory spike-train control.

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

## Validation and scope of this review

The retained scientific source files match their recorded SHA256 manifests. The package's CPU analysis and plotting commands were executed, including recomputation of Figure 5, Supplementary Table 6 and twelve Figure 6 data tables. These commands regenerate validation outputs under `outputs/verification/` and `outputs/analysis/`; generated outputs are excluded from Git.

The portable Linux build and simulation entry points were assembled from the validated production implementations and checked statically. A complete clean Linux/CUDA rebuild of this newly organized package has not yet been executed. Installation time and fresh-platform GPU runtimes therefore remain to be measured. Historical scientific validation is included in `data/validation/`; it is not a clean-install test of the packaging layer. A fresh reproduction must pass the included CPU/GPU/finite-difference gate before proceeding.

## Release and citation

The pre-publication repository is https://github.com/TiffanytorinoHe/S2E. It is private during manuscript preparation. Editors and reviewers can obtain access or a versioned code archive from the corresponding author. The initial review version is `v0.1.0-review`. Before publication, archive the publication version and frozen data in a DOI-minting repository, resolve the redistribution items in THIRD_PARTY.md, and update the manuscript with the permanent identifier. No DOI has yet been assigned.

`CITATION.cff` records the manuscript authors and title. Third-party attribution and outstanding redistribution checks are in [THIRD_PARTY.md](THIRD_PARTY.md); this archive does not relicense those assets.

The packaging follows the installation, example, validation and data-documentation items in the [Nature code submission guidance](https://media.nature.com/full/nature-cms/documents/GuidelinesCodePublication.pdf) and [computational reproducibility guidance](https://www.nature.com/documents/Computational_tools_reporting_guidelines.pdf). The target journal's submission requirements should be checked when selected.
