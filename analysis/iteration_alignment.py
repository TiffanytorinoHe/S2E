"""Verify selected iterative transfer/drive traces against the frozen solutions."""
from pathlib import Path
import json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1]
SNAPSHOT=ROOT/'data/provenance/publication_snapshot.json'
def read(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read_iteration_pack(path):
    path = Path(path).resolve()
    report = read(path.with_suffix(".json"))
    assert report.get("all_verified") and report["algorithm"] == "corrected_state_conditioned_v2"
    assert report["artifact_sha256"] == digest(path), "Iteration pack digest mismatch"
    with np.load(path, allow_pickle=False) as z:
        data = {key: z[key].copy() for key in z.files}
    if "pattern" in report:
        data = {"time_ms": data["time_ms"], **{report["pattern"]+"_"+key: data[key]
                for key in ["phi", "asyn", "steps"]}}
        sources = [report]
    else:
        sources = report["source_reports"]
    return data, report["rows"], sources

def load_iterations(path, sync_path=None):
    data, rows, sources = read_iteration_pack(path)
    provenance = [dict(path=str(Path(path).resolve()), sha256=digest(path),
                       patterns_used=sorted({r["pattern"] for r in rows}))]
    if sync_path is not None:
        sync, sync_rows, sync_sources = read_iteration_pack(sync_path)
        assert {r["pattern"] for r in sync_rows} == {"sync"}
        assert np.array_equal(data["time_ms"], sync["time_ms"])
        for field in ["phi", "asyn", "steps"]:
            data["sync_"+field] = sync["sync_"+field]
        rows = [r for r in rows if r["pattern"] != "sync"]+sync_rows
        sources = [r for r in sources if r["pattern"] != "sync"]+sync_sources
        provenance[0]["patterns_used"] = ["bhigh"]
        provenance.append(dict(path=str(Path(sync_path).resolve()), sha256=digest(sync_path),
                               patterns_used=["sync"]))
    snapshot = read(SNAPSHOT)
    frozen = {(g["key"], r["case"]): r for g in snapshot["groups"] for r in g["results"]}
    times = data["time_ms"]
    assert np.isfinite(times).all() and np.allclose(np.diff(times), .025, rtol=0, atol=1e-10)
    dt = float(np.diff(times)[0])
    cases, checks, sequence_lengths = {}, [], {}
    for pattern in ["bhigh", "sync"]:
        part = [r for r in rows if r["pattern"] == pattern]
        assert len(part) == 5 and len({r["case"] for r in part}) == 1
        case = cases[pattern] = int(part[0]["case"])
        source = [r for r in sources if r["pattern"] == pattern]
        assert len(source) == 1 and source[0]["case"] == case
        source = source[0]
        record = frozen["l5pc_"+pattern, case]
        result = record["result"]
        sequence_lengths[pattern] = len(result["ans"])
        assert source["all_verified"] and source["result_sha256"] == record["sha256"]
        assert source["final_replay_error_mV"] < 1e-8
        assert result["status"] == "complete" and result["final_spikes"] == 0 and result["final_max_mV"] < 0
        identity = result["identity"]
        release = identity.get("science_release_sha256", identity.get("release_sha256"))
        assert source["science_release_sha256"] == release
        assert source["input_sha256"] == identity["input_sha256"]
        steps = data[pattern+"_steps"]
        assert np.array_equal(steps, np.rint(np.linspace(0, len(result["ans"])-1, 5)).astype(int))
        assert [r["step"] for r in part] == steps.tolist()
        phi, drive = data[pattern+"_phi"], data[pattern+"_asyn"]
        assert phi.shape == drive.shape == (5, len(times))
        assert np.isfinite(phi).all() and np.isfinite(drive).all()
        history = {r["iteration"]: r for r in result["history"]}
        for i, row in enumerate(part):
            step = row["step"]
            assert row["candidate_id"] == result["ans"][step]
            assert row["segment"] == row["candidate_id"]//31
            assert row["onset_ms"] == 100+10*(row["candidate_id"] % 31)
            assert row["science_release_sha256"] == release and row["input_sha256"] == identity["input_sha256"]
            assert row["duality_error"] < 1e-8
            if step == 0:
                assert row["baseline_replay_error_mV"] < 1e-8
            direct = row["selected_score_mV_per_uS"]
            assert abs(direct-history[step]["score"])/max(1., abs(direct)) < 1e-8
            error = abs(np.dot(phi[i], drive[i])*dt-direct)/max(1., abs(direct))
            assert error < 1e-8
            assert np.max(abs(phi[i])) > 0 and np.max(abs(drive[i])) > 0
            checks.append(dict(pattern=pattern, case=case, step=step, signed_duality_error=error))
    return data, dict(algorithm="corrected_state_conditioned_v2", all_verified=True,
                      rows=rows, source_reports=sources, cases=cases,
                      sequence_lengths=sequence_lengths, checks=checks, provenance=provenance)
if __name__=='__main__':
 d=ROOT/'data/figures/figure4';data,report=load_iterations(d/'hf_iterations.npz',d/'sync_iterations.npz')
 out=ROOT/'outputs/analysis';out.mkdir(parents=True,exist_ok=True)
 (out/'iteration_alignment.json').write_text(json.dumps(report,indent=2))
 print('Verified',len(report['checks']),'selected traces and their signed inner products')
