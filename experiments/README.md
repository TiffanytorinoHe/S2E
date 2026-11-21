# Recomputing scientific results

Run commands from the repository root after building and sourcing `environment/activate.sh`. The group names and exact case lists are in `data/cohorts.json`; all sixteen model/condition groups contain fifty baseline-spiking realizations. Conditions are `ampa`, `nmda`, `bhigh` (HF NMDA/AMPA) and `sync`. CA1 group names use `ca1_20212`.

## Inhibitory search

```bash
python experiments/solve.py --group l5pc_ampa --case 0 --steps 15 --output outputs/demo/l5pc_ampa_000
```

The first invocation runs CPU/GPU/central-finite-difference checks in a separate process at zero and nonzero selected inhibition. Only a passing gate permits search. Repeating the identical command continues the checkpoint for up to fifteen additional selections. `solve.json` means the somatic maximum is below 0 mV; an unfinished checkpoint is not a completed solution. Repeated candidate choices remain repeated increments. Candidate `k` encodes segment `k//31` and onset `100 + 10*(k%31)` ms. Each selection adds 0.001 µS.

For the published L5 train control, add `--variant train` and use a separate output directory. The model preserves the perisomatic/distal train waveforms and plasticity mechanisms. Other-model train results are outside the published control and are rejected by this entry point.

The portable solver processes a bounded number of selections and preserves failures. Inspect `error.json` before using a new directory; no automatic failure retry or cluster submission is included. For a calibration protocol with a declared maximum count, an unfinished case at that count is censored, never a successful solution.

## Initial transfer–drive profiles

```bash
python experiments/profiles.py --group l5pc_ampa --case 0 --output outputs/demo/profile_l5pc_ampa_000
```

This collects the baseline (step-zero) complete-state tangent/adjoint profiles. It runs the same derivative gate first. Source-current signs, timestep factors and per-weight versus finite-increment scaling are defined in Supplementary Note 9 and the saved arrays. Spatial summaries use the publication definitions; Figure 4 absolute local-drive peaks and Figure 6 positive outward-drive peaks are intentionally different summaries.

## Autonomous voltage iteration

```bash
python experiments/voltage.py --group l5pc_ampa --case 0 --output outputs/demo/voltage_l5pc_ampa_000
```

The S2E state evolves from its own preceding voltage, gate and concentration values, using independently constructed synaptic waveforms. NEURON supplies initialization and the independent reference trajectory. `result.json` reports full-trajectory errors and spike times. The extra one-step diagnostic is not the voltage-validation result used by the paper.

## Reduced SIMPLE benchmark

```bash
python experiments/simple/reproduce.py --mode verify --case 26 --output outputs/demo/simple_026_verify
python experiments/simple/reproduce.py --mode solve --case 26 --output outputs/demo/simple_026_solve
python experiments/simple/reproduce.py --mode enumerate --case 26 --output outputs/demo/simple_026_enum
```

Enumeration is computationally expensive and uses the saved six-input realization with twenty inhibitory candidates. The supplied results contain 298 minima, with two length-eighteen cases (138 and 164) excluded from enumeration. The wrapper prevents silently adding them. Use this wrapper rather than the historical scientific module's standalone main block. All new files belong under `outputs/`; frozen data are read-only inputs.

No institution-specific scheduler, SSH helper, automatic successor or account credential is required by these commands. GPU assignment can be controlled externally with `CUDA_VISIBLE_DEVICES`, with one CUDA-owning process per GPU.

## Profile validation diagnostic (Supplementary Table 5)

```bash
python experiments/validate_profiles.py --group l5pc_ampa --output outputs/demo/diagnostic_l5
```

The retained diagnostic compares resident forward/adjoint kernels against the reference kernels, independent CPU responses and NEURON weight/current finite differences, then exports a 32-location × 31-onset profile block. The other reported cases are `ca1_20212_sync`, `l23_bhigh` and `msn_nmda`, each case 0. These commands use the current frozen input for that group. The original diagnostic measurements are retained in `data/validation/derivative_metrics.csv`.
