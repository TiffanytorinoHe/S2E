# Figure-to-code map

The six main figures supplied by the author are embedded unchanged in the companion manuscript. They are composited artwork. The numerical exports below are independent vector panels with the same frozen data; conceptual cartoons and final Illustrator/PowerPoint assembly are not regenerated.

| Paper item | Data | Recompute / draw |
|---|---|---|
| Fig. 1b | `figure1/voltage_trace.csv` | `experiments/voltage.py`; `main_panels.py` |
| Fig. 2c–e | `figure2/`; pulse solutions | `experiments/solve.py`; `main_panels.py` |
| Fig. 3a–f | `figure3/`; L5 pulse solutions | `analysis/verify_release.py`; `figure3.py` (b–d), `main_panels.py` (a,e,f) |
| Fig. 4c–f | `figure4/`; initial and selected-iteration profiles | `analysis/iteration_alignment.py`; `main_panels.py` |
| Fig. 5a–d | `figure5/`; 800 pulse solutions | `analysis/pulse_statistics.py`; `main_panels.py` |
| Fig. 6a–d | `figure6/`; 800 initial profiles | `analysis/transfer_drive.py`; `main_panels.py` |
| Supplementary Fig. 1 / Table 4 | `supplementary1/`; voltage validation | `experiments/voltage.py`; `supplementary.py` |
| Supplementary Fig. 2 | `supplementary2/`; SIMPLE results | `analysis/verify_release.py`; `supplementary.py` |
| Supplementary Fig. 3 | `supplementary3/`; example input and recording data | `supplementary.py` |
| Supplementary Fig. 4 / Table 6 | `supplementary4/`; 200 pulse/train pairs | `analysis/train_statistics.py`; `supplementary.py` |
| Supplementary Table 5 | `validation/derivative_metrics.csv` | Tangent/adjoint source and recorded diagnostic comparisons |

Data paths in this table are relative to `data/figures/` except those explicitly under `validation/`. Read the manuscript captions and Supplementary Note 9 for statistical units, normalization, bins, uncertainty and signs. Figure 2 pools selections within condition; Figure 5 weights realizations equally. Figure 5 colour limits are shared across conditions within each model. Geometry interpolation and branch thickness follow the main-figure morphology renderer. SVG text remains editable; PDF fonts are embedded as TrueType.

`figure3.py` includes Holm-corrected significance brackets versus Sync. Repeated inhibitory selections are retained. Sample sizes and explanatory details belong in the captions, not inside the data panels.
