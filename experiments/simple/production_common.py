from pathlib import Path
import hashlib,json,os
ROOT=Path(__file__).resolve().parent
RUN=ROOT.parents[1]/'data/simple'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic_json(path,data):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2));t.replace(p)
def inputs_sha():return {p.name:sha(p) for p in sorted((ROOT/'results_exc6/inputs').glob('*.npy'))}
def read_json(path):return json.loads(Path(path).read_text())
