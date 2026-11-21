"""Schema-2-only spatial theory analysis; historical data are read-only comparators.

No simulations, remote access, or implicit searches for profile inputs occur here.
The compact input contract is documented by ``input_contract`` below.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from read_profiles import read_profiles
from spatial_profile_processing import bin_mean_profile, positive_temporal_peak


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/analysis/transfer_drive"
MODELS = ["L5bPCmodelsEH", "CA1_20212", "L23", "MSN"]
REGIMES = ["ampa", "nmda", "bhigh", "sync"]
ALIASES = {"l5pc": MODELS[0], "ca1_20212": MODELS[1], "l23": MODELS[2], "msn": MODELS[3]}
BINS = [8, 16, 32]
THRESHOLDS = [.65, .75, .85]


def input_contract():
    return {
        "schema_version": 2,
        "baseline_step": 0,
        "state_conditioned": True,
        "cases": [{
            "model": "l5pc", "regime": "ampa", "case_id": "integer",
            "identity": {"group": "l5pc_ampa", "case": "integer",
                         "input_sha256": "sha256", "science_release_sha256": "sha256"},
            "kind": "segment_peaks or schema2_npz",
            "path": "path relative to manifest; npz or csv",
            "dt_ms": "positive timestep",
            "n_segments": "all non-axon locations; no sampling",
            "duality_error": "max |sum(K*a)-direct| / max(1,|direct|), <1e-8",
            "input_path": "optional local input path whose bytes match identity.input_sha256",
            "expected_identity": "optional independently checked full identity",
        }],
        "compact_columns_or_arrays": ["source_locations", "distance_um", "phi_peak", "a_peak", "score_peak"],
        "definitions": {
            "phi_peak": "max(0,max_t(-K_discrete/dt)); max(0,max_t(-K)) also accepted if phi_units=discrete",
            "a_peak": "max(0,max_t(a_syn_outward_nA_per_uS))",
            "score_peak": "max(0,-min_t(dV_s_dw_mV_per_uS)) for the selected onset",
            "coverage": "all_nonaxon",
            "phi_units": "density or discrete",
        },
        "requirements": [
            "Per-case baseline_step=0, state_conditioned=true (may be inherited from manifest).",
            "Per-case unique source_locations exactly 0..n_segments-1; finite distances and nonnegative peaks.",
            "Compact reduction must be downstream of schema-2 sign/unit/time/duality validation.",
            "Supply complete case inventory, source identities, and rejected/pending cases if available.",
            "Do not mix releases within one model/pattern. Document intentional cross-pattern calibration/release differences.",
            "Pair by model and case ID, not input hash: excitation schedules differ across patterns.",
        ],
    }


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    def clean(v):
        if isinstance(v, dict):
            return {str(k): clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        if isinstance(v, np.ndarray):
            return clean(v.tolist())
        if isinstance(v, np.generic):
            return clean(v.item())
        if isinstance(v, float) and not np.isfinite(v):
            return None
        return v
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding="utf-8")


def normalized(values):
    values = np.asarray(values, float)
    return values / values.sum() if values.sum() > 0 else np.zeros_like(values)


def location(profile, x, threshold=.75):
    profile = np.asarray(profile, float)
    if not profile.max() > 0:
        return np.nan
    use = profile >= threshold * profile.max()
    return float(np.average(np.asarray(x)[use], weights=profile[use]))


def bootstrap(values, seed=83):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    sample = v[rng.integers(0, len(v), (4000, len(v)))].mean(axis=1)
    return float(v.mean()), *np.percentile(sample, [2.5, 97.5]).tolist()


def _identity(entry, manifest):
    identity = entry["identity"]
    if entry.get("baseline_step", manifest.get("baseline_step")) != 0:
        raise ValueError("Only unmodified step-0 baseline profiles are permitted")
    if entry.get("state_conditioned", manifest.get("state_conditioned")) is not True:
        raise ValueError("Missing state-conditioned baseline declaration")
    model = ALIASES.get(entry["model"], entry["model"])
    regime = entry.get("regime", entry.get("pattern"))
    case = int(entry.get("case_id", entry.get("case", identity.get("case"))))
    if model not in MODELS or regime not in REGIMES or identity.get("case") != case:
        raise ValueError("Invalid model/pattern/case identity")
    group = next(k for k, v in ALIASES.items() if v == model) + "_" + regime
    if identity.get("group") != group:
        raise ValueError(f"Group identity mismatch: {identity.get('group')} != {group}")
    release = identity.get("science_release_sha256", identity.get("release_sha256"))
    for name, value in [("input_sha256", identity.get("input_sha256")), ("release", release)]:
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value.lower()):
            raise ValueError(f"Missing or invalid {name}")
    if "expected_identity" in entry and entry["expected_identity"] != identity:
        raise ValueError("Expected scientific identity mismatch")
    return model, regime, case, release


def agent2_manifest(manifest, manifest_path, snapshot_path, require_complete=True):
    """Bind compact arrays to the independent frozen producer/input inventory."""
    snapshot_path = Path(snapshot_path).resolve()
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8-sig"))
    anchor = sha256(snapshot_path)
    if manifest["snapshot_sha256"] != anchor:
        raise ValueError("Compact summary and frozen snapshot hashes differ")
    if manifest["errors"]:
        raise ValueError("Compact reduction reported errors; cannot treat it as a complete cohort")
    expected = {(p["group"], p["case"]): p for p in snapshot["profiles"]}
    actual = {(p["group"], p["case"]): p for p in manifest["profiles"] if p["role"] == "formal_cohort"}
    if len(actual) != sum(p["role"] == "formal_cohort" for p in manifest["profiles"]):
        raise ValueError("Duplicate compact formal case")
    if not set(actual) <= set(expected) or (require_complete and set(actual) != set(expected)):
        raise ValueError(f"Frozen profile inventory mismatch: compact={len(actual)}, expected={len(expected)}")
    groups = {g["key"]: g for g in snapshot["manifest"]["groups"]}
    records = {g["key"]: g for g in snapshot["groups"]}
    release_audit = []
    for model in ALIASES:
        model_groups = [g for g in groups.values() if g["model"] == model]
        hashes = {g["release_sha256"] for g in model_groups}
        manifests = [r["data"] for r in snapshot["releases"].values() if r["sha256"] in hashes]
        common = set.intersection(*(set(r) for r in manifests))
        invariant = {k: manifests[0][k] for k in common if len({r[k] for r in manifests}) == 1}
        changed = sorted(k for k in set.union(*(set(r) for r in manifests)) if len({r.get(k) for r in manifests}) > 1)
        if model == "msn":
            required = ["MSN.py", "params_dMSN.json", "latest_WT-P270-20-14ak.swc", "model_config.json"]
            required += [k for k in common if k.startswith("mod/")]
            if any(k not in invariant for k in required):
                raise ValueError("MSN morphology/biophysics/mechanism sources differ across conditions")
        release_audit.append(dict(model=ALIASES[model], n_releases=len(hashes),
                                  group_configuration=[{k: g.get(k) for k in ["key", "release_sha256", "science_root", "new_msn_config", "counts", "hf_clusters"]} for g in model_groups],
                                  common_invariant_source_hashes=invariant, differing_release_members=changed,
                                  interpretation="MSN pattern comparisons include calibrated input-count changes and distinct run packages, not a pure pattern-only perturbation." if len(hashes) > 1 else "One scientific release across patterns."))
    entries = []
    for key, item in actual.items():
        reference = expected[key]
        path = manifest_path.parent / item["path"]
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8-sig"))
        report = reference["report"]["data"]
        complete = reference["complete"]["data"]
        identity = report["identity"]
        if meta["identity"] != identity or identity != complete["identity"]:
            raise ValueError(f"Producer/summary identity mismatch: {key}")
        if meta["snapshot_sha256"] != anchor or meta["report_sha256"] != reference["report"]["sha256"]:
            raise ValueError(f"Producer/snapshot provenance mismatch: {key}")
        if meta["artifact"] != reference["artifact"] or meta["source_npz_sha256"] != reference["source_sha256"]:
            raise ValueError(f"Source artifact mismatch: {key}")
        if sha256(path) != item["sha256"] or item["sha256"] != meta["summary_sha256"]:
            raise ValueError(f"Compact checksum mismatch: {key}")
        if meta["source_bytes"] != reference["bytes"] or meta["source_mtime_ns"] != reference["mtime_ns"]:
            raise ValueError(f"Source census changed: {key}")
        if complete["baseline_max_abs_difference_mV"] != 0 or complete["baseline_spikes"] <= 0:
            raise ValueError(f"Unverified baseline identity or nonspiking baseline: {key}")
        group = groups[key[0]]
        inp = next(i for i in group["inputs"] if i["case"] == key[1])
        if inp["sha256"] != identity["input_sha256"] or group["release_sha256"] != identity["science_release_sha256"]:
            raise ValueError(f"Frozen input/release mismatch: {key}")
        geometry = records[key[0]]["geometry"]
        n = geometry["nonaxon_segments"]
        if n != report["segments"] or n != meta["segments"] or n != item["segments"]:
            raise ValueError(f"Incomplete non-axon geometry: {key}")
        entries.append(dict(model=group["model"], regime=group["regime"], case_id=key[1],
                            identity=identity, expected_identity=complete["identity"],
                            kind="agent2_compact", path=item["path"], n_segments=n,
                            baseline_step=report["baseline_step"], state_conditioned=report["state_conditioned"],
                            input_path=str(ROOT / "data/inputs" / key[0] / f"case_{key[1]:03d}.npz"),
                            duality_error=meta["duality_error"], expected_distance_um=geometry["distance_um"][:n],
                            baseline_spikes=complete["baseline_spikes"], producer_metadata=meta))
    return dict(schema_version=2, baseline_step=0, state_conditioned=True, coverage="all_nonaxon",
                cases=entries, snapshot_path=str(snapshot_path), snapshot_sha256=anchor,
                snapshot_cutoff_utc=datetime.fromtimestamp(snapshot["unix"], tz=timezone.utc).isoformat(),
                frozen_profile_count=len(expected), completeness_required=require_complete,
                compact_reduction_script_sha256=manifest["script_sha256"],
                cross_pattern_release_audit=release_audit,
                noncohort_diagnostics=[p for p in manifest["profiles"] if p["role"] != "formal_cohort"],
                exclusion_definition="Diagnostic/report-only artifacts are outside the formal completed cohort; they may duplicate cases or sample only some locations.",
                baseline_identity="Every formal producer reports exactly zero baseline voltage difference from the input run; compact and completion identities match; local input bytes rehashed.",
                frozen_nonspiking_and_pending={g["key"]: {k: g[k] for k in ["nonspiking", "pending"]} for g in snapshot["groups"]})


def load_manifest(manifest_path, snapshot_path=None, require_complete=True):
    """Load explicitly supplied raw or validated compact schema-2 profiles."""
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if "profiles" in manifest and "snapshot_sha256" in manifest:
        if snapshot_path is None:
            raise ValueError("Agent2 summaries require the independently frozen --snapshot")
        manifest = agent2_manifest(manifest, manifest_path, snapshot_path, require_complete)
    if manifest.get("schema_version") != 2:
        raise ValueError("Explicit schema_version=2 is required")
    cases, audit, seen, geometries, releases = [], [], set(), {}, {}
    for entry in manifest["cases"]:
        model, regime, case, release = _identity(entry, manifest)
        key = (model, regime, case)
        if key in seen:
            raise ValueError(f"Duplicate profile case: {key}")
        seen.add(key)
        releases.setdefault((model, regime), release)
        if releases[model, regime] != release:
            raise ValueError(f"Mixed scientific releases within {model}/{regime}")
        source = (manifest_path.parent / entry["path"]).resolve()
        if "input_path" in entry:
            if sha256(manifest_path.parent / entry["input_path"]) != entry["identity"]["input_sha256"]:
                raise ValueError(f"Input bytes mismatch for {key}")
        kind = entry.get("kind", "segment_peaks")
        if kind == "schema2_npz":
            sidecar = json.loads(source.with_suffix(".json").read_text(encoding="utf-8-sig"))
            if sidecar.get("baseline_step") != 0 or sidecar.get("state_conditioned") is not True:
                raise ValueError("Raw schema-2 baseline report mismatch")
            raw = read_profiles(source, expected_identity=entry["identity"])
            if not np.array_equal(raw["target_steps"], np.argmin(raw["dV_s_dw"], axis=0)):
                raise ValueError("Direct score target is not the selected-onset minimum")
            values = dict(source_locations=raw["source_locations"], distance_um=raw["distance_um"],
                          phi_peak=positive_temporal_peak(raw["phi_density"], axis=0),
                          a_peak=positive_temporal_peak(raw["a_syn_outward"], axis=0),
                          score_peak=positive_temporal_peak(-raw["dV_s_dw"], axis=0))
            dt, duality = raw["dt_ms"], raw["duality_error"]
        elif kind == "agent2_compact":
            with np.load(source, allow_pickle=False) as z:
                dt, duality = float(z["dt_ms"]), float(entry["duality_error"])
                values = dict(source_locations=z["source_locations"].copy(), distance_um=z["distance_um"].copy(),
                              phi_peak=np.maximum(z["phi_density_peak"], 0),
                              a_peak=np.maximum(z["a_peak"], 0), score_peak=np.maximum(z["score_mV_per_uS"], 0))
                if not np.allclose(z["phi_density_peak"], z["phi_peak"]/dt, rtol=1e-12, atol=1e-12):
                    raise ValueError("Compact discrete/density transfer scaling mismatch")
                score = z["score_mV_per_uS"]
                reconstructed = z["reconstructed_score_mV_per_uS"]
                err = float(np.max(abs(score-reconstructed)/np.maximum(1., abs(score))))
                if err >= 1e-8 or not np.isclose(err, duality, rtol=1e-5, atol=1e-13):
                    raise ValueError("Compact signed response duality mismatch")
                if not np.array_equal(z["onset_indices"], np.argmax(z["score_by_onset_mV_per_uS"], axis=1)):
                    raise ValueError("Compact selected onset mismatch")
                if not np.allclose(score, z["score_by_onset_mV_per_uS"].max(axis=1), rtol=1e-10, atol=1e-10):
                    raise ValueError("Compact target score mismatch")
                denom = z["phi_density_L2"]*z["a_L2"]
                alignment = np.divide(reconstructed, denom, out=np.full_like(score, np.nan), where=denom > 0)
                if not np.allclose(alignment, z["signed_alignment"], rtol=1e-12, atol=1e-12, equal_nan=True):
                    raise ValueError("Compact temporal alignment identity mismatch")
                if np.any(abs(alignment[np.isfinite(alignment)]) > 1+1e-10):
                    raise ValueError("Signed temporal alignment violates Cauchy-Schwarz")
                if int(z["baseline_spike_count"]) != entry["baseline_spikes"]:
                    raise ValueError("Compact baseline spike count mismatch")
                if not np.allclose(values["distance_um"], entry["expected_distance_um"], rtol=0, atol=1e-8):
                    raise ValueError("Compact source geometry differs from frozen input geometry")
        elif kind == "segment_peaks":
            if entry.get("validated_schema2", manifest.get("validated_schema2")) is not True:
                raise ValueError("Compact reductions need an explicit schema-2 validation declaration")
            if source.suffix.lower() == ".npz":
                with np.load(source, allow_pickle=False) as z:
                    values = {name: z[name].copy() for name in input_contract()["compact_columns_or_arrays"]}
            elif source.suffix.lower() == ".csv":
                table = pd.read_csv(source)
                if "case_id" in table:
                    table = table[table.case_id == case]
                if "model" in table:
                    table = table[table.model.isin([model, entry["model"]])]
                if "regime" in table:
                    table = table[table.regime == regime]
                values = {name: table[name].to_numpy() for name in input_contract()["compact_columns_or_arrays"]}
            else:
                raise ValueError("Compact profile path must be npz or csv")
            dt, duality = float(entry["dt_ms"]), float(entry["duality_error"])
            units = entry.get("phi_units", manifest.get("phi_units"))
            if units not in ["discrete", "density"]:
                raise ValueError("Declare compact phi_units as discrete or density")
            if units == "discrete":
                values["phi_peak"] = values["phi_peak"] / dt
        else:
            raise ValueError(f"Unsupported profile kind: {kind}")
        if not np.isfinite(dt) or dt <= 0 or not np.isfinite(duality) or not 0 <= duality < 1e-8:
            raise ValueError("Invalid timestep or kernel/current/response duality")
        n = int(entry["n_segments"])
        if entry.get("coverage", manifest.get("coverage")) != "all_nonaxon":
            raise ValueError("Full non-axon location coverage must be declared")
        if any(np.asarray(v).shape != (n,) for v in values.values()):
            raise ValueError(f"Misaligned compact arrays for {key}")
        if not all(np.isfinite(v).all() for v in values.values()):
            raise ValueError(f"Nonfinite compact values for {key}")
        order = np.argsort(values["source_locations"])
        values = {k: np.asarray(v)[order] for k, v in values.items()}
        if not np.array_equal(values["source_locations"], np.arange(n)):
            raise ValueError(f"Incomplete or duplicate location coverage for {key}")
        distance = values["distance_um"]
        if np.any(distance < 0) or distance.max() <= 0:
            raise ValueError("Invalid soma-path distance")
        if any(np.any(values[q] < 0) for q in ["phi_peak", "a_peak", "score_peak"]):
            raise ValueError("Compact arrays must contain positive temporal peaks")
        geometries.setdefault(model, distance)
        if distance.shape != geometries[model].shape or not np.allclose(distance, geometries[model], rtol=0, atol=1e-8):
            raise ValueError(f"Inconsistent per-model geometry: {key}")
        cases.append(dict(model=model, regime=regime, seed=case, **values))
        audit.append(dict(model=model, regime=regime, case_id=case, source=str(source),
                          source_sha256=sha256(source), identity=entry["identity"],
                          baseline_step=0, n_segments=n, dt_ms=dt, duality_error=duality,
                          zero_peak_locations={q: int(np.count_nonzero(values[q] == 0)) for q in ["phi_peak", "a_peak", "score_peak"]},
                          input_bytes_checked="input_path" in entry,
                          expected_identity_checked="expected_identity" in entry,
                          producer_metadata=entry.get("producer_metadata")))
    if not cases:
        raise ValueError("No completed profile cases supplied")
    return cases, dict(manifest=str(manifest_path), manifest_sha256=sha256(manifest_path),
                       cases=audit, source_metadata={k: v for k, v in manifest.items() if k != "cases"})


def build_profiles(cases):
    rows = []
    for case in cases:
        distance = case["distance_um"]
        for nb in BINS:
            for quantity, field in [("phi", "phi_peak"), ("a", "a_peak"), ("score", "score_peak")]:
                Q, q, x, edges = bin_mean_profile(case[field], distance, n_bins=nb)
                for i in range(nb):
                    rows.append(dict(model=case["model"], regime=case["regime"], seed=case["seed"],
                                     bins=nb, quantity=quantity, bin=i, distance_um=x[i],
                                     distance_norm=x[i]/distance.max(), max_distance_um=distance.max(),
                                     bin_left_um=edges[i], bin_right_um=edges[i+1], Q=Q[i], q=q[i],
                                     peak_norm=Q[i]/Q.max() if Q.max() > 0 else np.nan))
    return pd.DataFrame(rows)


def cohort_counts(cases):
    rows, paired, complete, seeds, all_ids = [], [], {}, {}, {}
    for model in MODELS:
        sets = {r: {c["seed"] for c in cases if c["model"] == model and c["regime"] == r} for r in REGIMES}
        for r in REGIMES:
            rows.append(dict(model=model, regime=r, n_cases=len(sets[r]), case_ids=sorted(sets[r])))
        intersections = []
        for r in REGIMES[1:]:
            ids = sets["ampa"] & sets[r]
            intersections.append(ids)
            paired.append(dict(model=model, regime=r, baseline="ampa", n_pairs=len(ids), case_ids=sorted(ids),
                               unmatched_current_case_ids=sorted(sets[r]-sets["ampa"]),
                               unmatched_baseline_case_ids=sorted(sets["ampa"]-sets[r])))
        complete[model] = sorted(set.intersection(*sets.values()))
        seeds[model] = sorted(set.union(*intersections))
        all_ids[model] = sorted(set.union(*sets.values()))
    return dict(total_profile_cases=len(cases), per_pattern=rows, baseline_intersections=paired,
                per_pattern_counts={m: {r["regime"]: r["n_cases"] for r in rows if r["model"] == m} for m in MODELS},
                matched_transition_counts={m: {r["regime"]: r["n_pairs"] for r in paired if r["model"] == m} for m in MODELS},
                all_pattern_intersection_case_ids=complete,
                all_pattern_intersection_counts={m: len(s) for m, s in complete.items()},
                paired_case_union_case_ids=seeds, paired_case_union_counts={m: len(s) for m, s in seeds.items()},
                all_case_id_union_counts={m: len(s) for m, s in all_ids.items()},
                total_paired_transitions=sum(r["n_pairs"] for r in paired),
                display_cohort="Every available profile case separately within each model/input pattern; no downsampling.",
                dependent_metric_cohort="Each non-AMPA pattern intersected with AMPA by model and case ID; all available such pairs.",
                pooled_estimand="Mean available transition values within case, then equal-weight mean over cases; pattern availability may differ.",
                pairing_limitations=[
                    "Case-ID matching is an operational repeated-realization pairing, not evidence of identical excitation schedules across input patterns.",
                    "Input hashes are checked within a source case, not required to agree across patterns.",
                    "Bootstrap units are simulated input cases within one morphology, not biological replicates.",
                    "Complete-four-pattern intersections are documented and exported as sensitivity analyses, not used to discard other available cases.",
                ])


def analyze(profiles, counts):
    cache = {}
    for key, group in profiles.groupby(["model", "regime", "seed", "bins"]):
        arrays = {q: g.sort_values("bin").q.to_numpy() for q, g in group.groupby("quantity")}
        arrays["x"] = group[group.quantity == "phi"].sort_values("bin").distance_norm.to_numpy()
        cache[key] = arrays
    elastic, hybrids, validation, baseline = [], [], [], []
    max_efficiency_error = 0.
    for (model, regime, seed, nb), cur in cache.items():
        product = normalized(cur["phi"] * cur["a"])
        corr = float(np.corrcoef(product, cur["score"])[0, 1]) if np.std(product) > 0 and np.std(cur["score"]) > 0 else np.nan
        z_proxy, z_direct = location(product, cur["x"]), location(cur["score"], cur["x"])
        validation.append(dict(model=model, regime=regime, seed=seed, bins=nb, correlation=corr,
                               proxy_location=z_proxy, direct_location=z_direct,
                               abs_location_error=abs(z_proxy-z_direct)))
        if regime == "ampa":
            self_phi = np.linalg.norm(cur["phi"]-cur["phi"])/np.linalg.norm(cur["phi"]) if np.linalg.norm(cur["phi"]) > 0 else np.nan
            self_a = np.linalg.norm(cur["a"]-cur["a"])/np.linalg.norm(cur["a"]) if np.linalg.norm(cur["a"]) > 0 else np.nan
            z = [location(normalized(cur["phi"]*cur["a"]), cur["x"]) for _ in range(4)]
            self_cp, self_ca = .5*((z[1]-z[0])+(z[3]-z[2])), .5*((z[2]-z[0])+(z[3]-z[1]))
            if any(np.isfinite(v) and v != 0 for v in [self_phi, self_a, self_cp, self_ca]):
                raise AssertionError("Nonzero self-transition metric")
            baseline.append(dict(model=model, seed=seed, bins=nb,
                                 E_phi_self=self_phi, E_a_self=self_a, c_phi_self=self_cp, c_a_self=self_ca,
                                 product_defined=bool(product.sum() > 0)))
            continue
        base = cache.get((model, "ampa", seed, nb))
        if base is None:
            continue
        if not np.allclose(base["x"], cur["x"], rtol=0, atol=1e-12):
            raise ValueError("Paired spatial grids differ")
        row = dict(model=model, regime=regime, seed=seed, bins=nb)
        for q in ["phi", "a"]:
            valid = np.linalg.norm(base[q]) > 0 and np.linalg.norm(cur[q]) > 0
            row["E_"+q] = float(np.linalg.norm(cur[q]-base[q])/np.linalg.norm(base[q])) if valid else np.nan
            row["E_"+q+"_peaknorm"] = float(np.linalg.norm(cur[q]/cur[q].max()-base[q]/base[q].max())/np.linalg.norm(base[q]/base[q].max())) if valid else np.nan
        elastic.append(row)
        hh = [normalized(base["phi"]*base["a"]), normalized(cur["phi"]*base["a"]),
              normalized(base["phi"]*cur["a"]), normalized(cur["phi"]*cur["a"])]
        for threshold in THRESHOLDS:
            z00, z10, z01, z11 = [location(h, cur["x"], threshold) for h in hh]
            cp = .5*((z10-z00)+(z11-z01))
            ca = .5*((z01-z00)+(z11-z10))
            if np.isfinite(cp+ca):
                err = abs(cp+ca-(z11-z00))
                if err > 1e-12:
                    raise AssertionError("Shapley efficiency failed")
                max_efficiency_error = max(max_efficiency_error, err)
            denom = abs(cp)+abs(ca)
            fraction = abs(cp)/denom if np.isfinite(denom) and denom > 1e-10 else np.nan
            hybrids.append(dict(model=model, regime=regime, seed=seed, bins=nb, threshold=threshold,
                                z00=z00, z10=z10, z01=z01, z11=z11, c_phi=cp, c_a=ca,
                                abs_phi=abs(cp), abs_a=abs(ca), R_phi=fraction, proxy_shift=z11-z00,
                                direct_shift=location(cur["score"], cur["x"], threshold)-location(base["score"], cur["x"], threshold),
                                undefined_fraction=not np.isfinite(fraction)))
    if not elastic:
        raise ValueError("No same-case AMPA transitions available; dependent panels cannot be generated")
    es, hs, vs = pd.DataFrame(elastic), pd.DataFrame(hybrids), pd.DataFrame(validation)
    em = es.groupby(["model", "seed", "bins"]).agg(
        E_phi=("E_phi", "mean"), E_a=("E_a", "mean"),
        E_phi_peaknorm=("E_phi_peaknorm", "mean"), E_a_peaknorm=("E_a_peaknorm", "mean"),
        n_regimes=("regime", "nunique"), n_defined_phi=("E_phi", "count")).reset_index()
    main = hs[(hs.bins == 8) & np.isclose(hs.threshold, .75)].copy()
    sm = main.groupby(["model", "seed"]).agg(abs_phi=("abs_phi", "mean"), abs_a=("abs_a", "mean"),
                                              n_regimes=("regime", "nunique"), n_defined=("abs_phi", "count")).reset_index()
    sm["paired_contrast"] = sm.abs_phi-sm.abs_a
    summary, e_summary, regime_summary = [], [], []
    for model in MODELS:
        g = sm[sm.model == model]
        row = dict(model=model, n_seed=len(g), n_defined=int(g.paired_contrast.notna().sum()))
        for q in ["abs_phi", "abs_a"]:
            row[q], row[q+"_ci_low"], row[q+"_ci_high"] = bootstrap(g[q])
        row["mean_difference"], row["difference_ci_low"], row["difference_ci_high"] = bootstrap(g.paired_contrast)
        if np.isfinite(row["mean_difference"]):
            np.testing.assert_allclose(row["mean_difference"], row["abs_phi"]-row["abs_a"], rtol=0, atol=1e-12)
        row["ratio_of_means"] = row["abs_phi"]/(row["abs_phi"]+row["abs_a"]) if row["abs_phi"]+row["abs_a"] > 0 else np.nan
        summary.append(row)
        ev = em[(em.model == model) & (em.bins == 8)].E_phi
        mu, lo, hi = bootstrap(ev)
        e_summary.append(dict(model=model, n_seed=len(ev), n_defined=int(ev.notna().sum()), E_phi=mu, ci_low=lo, ci_high=hi))
        for regime in REGIMES[1:]:
            g = main[(main.model == model) & (main.regime == regime)]
            mu, lo, hi = bootstrap(g.R_phi)
            delta, dlo, dhi = bootstrap(g.abs_phi-g.abs_a)
            eg = es[(es.model == model) & (es.regime == regime) & (es.bins == 8)]
            e_mu, e_lo, e_hi = bootstrap(eg.E_phi)
            regime_summary.append(dict(model=model, regime=regime, n=len(g), n_defined=int(g.R_phi.notna().sum()),
                                       R_phi=mu, ci_low=lo, ci_high=hi, abs_phi_mean=g.abs_phi.mean(), abs_a_mean=g.abs_a.mean(),
                                       mean_difference=delta, difference_ci_low=dlo, difference_ci_high=dhi,
                                       E_phi=e_mu, E_phi_ci_low=e_lo, E_phi_ci_high=e_hi))
    complete_e = es[[s in counts["all_pattern_intersection_case_ids"][m] for m, s in zip(es.model, es.seed)]]
    complete_h = main[[s in counts["all_pattern_intersection_case_ids"][m] for m, s in zip(main.model, main.seed)]]
    complete_e = complete_e.groupby(["model", "seed", "bins"])[["E_phi", "E_a"]].mean().reset_index()
    complete_h = complete_h.groupby(["model", "seed"])[["abs_phi", "abs_a"]].mean().reset_index()
    tables = {
        "profiles": profiles, "elasticity_by_transition": es, "shapley_proxy_by_transition": hs,
        "proxy_validation": vs, "baseline_self_checks": pd.DataFrame(baseline),
        "elasticity_seed_means": em, "absolute_contributions_seed_means": sm,
        "absolute_contribution_summary": pd.DataFrame(summary), "ephi_summary": pd.DataFrame(e_summary),
        "shapley_regime_summary": pd.DataFrame(regime_summary),
        "complete_case_elasticity_seed_means": complete_e,
        "complete_case_contributions_seed_means": complete_h,
    }
    audit = dict(max_shapley_efficiency_error=max_efficiency_error,
                 undefined_main_fraction_count=int(main.undefined_fraction.sum()),
                 undefined_main_contribution_count=int(main.c_phi.isna().sum()),
                 undefined_E_phi_count=int(es[es.bins == 8].E_phi.isna().sum()),
                 baseline_self_check_rows=len(baseline),
                 profile_validation_8bins=vs[vs.bins == 8].groupby("model").agg(
                     n_cases=("seed", "size"), median_correlation=("correlation", "median"),
                     median_abs_location_error=("abs_location_error", "median"),
                     n_defined_locations=("abs_location_error", "count")).to_dict("index"),
                 definitions={
                     "segment_peak": "Positive temporal maximum of paper-sign transfer (-K/dt), outward local drive, and -dV_s/dw.",
                     "drive_sign_equivalence": "A=max(0,max_t(a_outward))=max(0,-min_t(didp_syn)/D1) for didp_syn=-a_outward*D1, D1>0 constant in time. This is not max(abs(a_outward)).",
                     "conceptual_signed_law": "With phi=-K/dt and a_syn_signed=-a_outward, dV_s/dw=dt*sum(phi*a_syn_signed). The positive spatial amplitude A is not substituted into this signed temporal equation.",
                     "binning": "Unweighted mean across all non-axon segments in equal-width bins spanning min to max distance; empty bins are zero.",
                     "q": "Q / (sum(Q) + 1e-12), exactly the existing spatial_profile_processing.bin_mean_profile convention.",
                     "E_phi": "||q_phi(current)-q_phi(AMPA)||_2 / ||q_phi(AMPA)||_2; 8 bins; zero-total profiles flagged undefined.",
                     "display": "16 bins; average all available per-pattern case q vectors, then divide each displayed curve by its own maximum.",
                     "surrogate": "H_ij = normalize(q_phi(i)*q_a(j)); not the exact temporal inner product.",
                     "location": "H-weighted mean of normalized bin-center distances with H >= 0.75*max(H).",
                     "Shapley": "c_phi=((z10-z00)+(z11-z01))/2; c_a=((z01-z00)+(z11-z10))/2.",
                     "aggregation": "Absolute contributions per transition, mean across available paired patterns within case, then mean across cases.",
                     "contrast": "Paired case-level abs_phi-abs_a; exactly equals difference of bar means when defined.",
                     "bootstrap": "4000 case resamples, random seed 83, percentile 95% CI; no independent transition resampling.",
                 }, assumptions=[
                     "Schema-2 transfer is target- and baseline-state-conditioned; selected onset and target can differ by location and pattern.",
                     "Exact signed law is -dV_s/dw = sum((-K)*a) = dt*sum((-K/dt)*a).",
                     "Taking separate positive temporal peaks, bin averaging, and multiplying spatial profiles do not commute with the exact temporal inner product.",
                     "A location-displacement decomposition is not a decomposition of finite-amplitude voltage or inhibitory selection frequency.",
                     "Undefined zero/zero contribution fractions remain NaN; cancellation of two nonzero contributions is not zero/zero.",
                     "Architecture descriptions refer to four model morphologies; no universal classifier or causal architectural effect is inferred.",
                 ])
    return tables, audit


def conclusions(tables):
    return {"cohort": "Frozen publication inputs and all 800 initial profiles"}


def run_analysis(manifest_path, snapshot_path=None):
    cases, source = load_manifest(manifest_path, snapshot_path)
    counts = cohort_counts(cases)
    counts["snapshot_cutoff_utc"] = source["source_metadata"].get("snapshot_cutoff_utc")
    counts["all_frozen_formal_cases_included"] = len(cases) == source["source_metadata"].get("frozen_profile_count", len(cases))
    counts["cohort_definition"] = "Every formal completed all-location schema-2 step-0 profile in the frozen inventory, independently of whether a later inhibition optimization finished."
    counts["pairing_limitations"].append("MSN AMPA, NMDA/HF, and Sync use three documented release packages and calibrated input configurations. Intrinsic morphology, parameter, and channel-mechanism source hashes match, but pattern comparisons are not a pure pattern-only intervention.")
    tables, math = analyze(build_profiles(cases), counts)
    for name in ["source_data", "metadata", "qa"]:
        (OUT / name).mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(OUT / "source_data" / (name+".csv"), index=False)
    save_json(OUT / "metadata/cohort_counts.json", counts)
    save_json(OUT / "metadata/source_audit.json", source)
    save_json(OUT / "metadata/cross_pattern_release_audit.json", source["source_metadata"].get("cross_pattern_release_audit", []))
    save_json(OUT / "metadata/mathematical_audit.json", math)
    save_json(OUT / "metadata/conclusions.json", conclusions(tables))
    return tables, counts

if __name__=='__main__':
    run_analysis(ROOT/'data/profiles/summary_manifest.json',ROOT/'data/provenance/publication_snapshot.json')
