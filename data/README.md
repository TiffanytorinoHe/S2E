# Data and provenance

## Included cohorts

| Data | Scope |
|---|---|
| `inputs/`, `cohorts.json` | 800 fixed inputs, 50 baseline-spiking realizations in each of 16 groups |
| `solutions/pulse/` | All 800 completed pulse solutions, including repeated increments and terminal voltage checks |
| `profiles/formal/` | 800 compact initial transfer–drive profiles and metadata |
| `validation/` | Autonomous-voltage results for 800 cases and derivative validation records |
| `figures/supplementary4/verified_results.json` | 200 paired L5 pulse/train results, fifty per excitation condition |
| `simple/` | 300 algorithm results, 298 enumeration results, CPU certificates and audits |
| `figures/` | Main and supplementary plot data, representative voltage and iterative kernel traces |
| `geometry/` | Segment morphology, path distances and anatomical region labels |

MSN uses the final AMPA configuration (2160/2160/600 AMPA/NMDA/background-GABA entries), calibrated NMDA (480/480/150), calibrated HF (320/320/100) and original Sync inputs. These are separate scientific conditions; earlier calibration candidates are not pooled into the publication cohort. All original source identities remain available in `provenance/`.

The detailed simulations use 0.025 ms updates and run to 400.025 ms with native model temperature and the recorded initialization. HH tables are disabled. Native model segmentation and geometry are retained. Data in this package must not be relabelled as results of the separate growing-neuron mesh protocol.

## Identity records

Scientific runtime manifests are stored with each model. Input archive SHA256 values are in `cohorts.json`. During cohort expansion, 144 input archives were repacked. For these cases, `provenance/original_inputs/` retains the exact original archives named by the result hashes; `input_equivalence.json` records array-by-array simulation identity with the packaged inputs. This distinction preserves both archival byte identity and the unchanged scientific inputs.

The raw SIMPLE certificate tree is retained under `simple/cpu_archive/`. Cases 0 and 26 keep their `validation/` paths. `provenance/simple_cpu_certificates.json` maps all 298 cases to these certificates and gives hashes. Cases 138 and 164 have algorithm results but no claimed enumerated minimum. Of the 298 enumerated cases, 64 were silent at baseline; the publication comparison uses the remaining 234.

Some original JSON records contain historical absolute paths and dates as provenance. The portable commands do not use those server paths to connect or write. Renaming these provenance fields would break their role as records of the original experiment.

## Retention and public release

The review archive includes fixed inputs, model assets, final solutions, compact profiles, representative traces, exact figure data and validation evidence. It excludes transient cluster logs, checkpoint duplicates, unsuccessful calibration sweeps and full all-compartment trajectories that can be regenerated from the fixed inputs. No experimental data are deleted from the working archive.

For public release, keep source code in GitHub and archive this frozen dataset with a persistent DOI, with checksums and a direct link from the README. Avoid committing large binary data repeatedly to Git history. `MANIFEST.sha256` provides a package-wide identity check. The `data/` bundle is necessary for exact reproduction; source code alone does not reproduce the fixed cohort.
