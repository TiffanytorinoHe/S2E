"""Recheck frozen identities and recompute the main numerical summaries on CPU."""
from pathlib import Path
import hashlib,json,platform,sys
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu,spearmanr
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'data';OUT=ROOT/'outputs/verification'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def gini(x):
 x=np.asarray(x,float);x=np.sort(x)
 return float(2*np.dot(np.arange(1,len(x)+1),x)/(len(x)*x.sum())-(len(x)+1)/len(x)) if x.sum() else np.nan
def main():
 OUT.mkdir(parents=True,exist_ok=True);rows=[];groups=read(D/'cohorts.json')['groups']
 assert len(groups)==16
 metrics=pd.read_csv(D/'figures/figure5/case_metrics.csv').set_index(['model','pattern','case_id'])
 equivalents={(q['group'],q['case']):q for q in read(D/'provenance/input_equivalence.json')}
 for g in groups:
  assert len(g['selected'])==len(set(g['selected']))==50
  inputs={i['case']:i for i in g['inputs']};assert set(inputs)==set(g['selected'])
  s=read(D/'solutions/pulse'/f"{g['key']}.json")
  records={r['case']:r for r in s['results']};assert len(records)==len(s['results'])==50
  assert set(records)==set(g['selected'])
  distances=np.asarray(s['geometry']['distance_um'])
  for case in g['selected']:
   inp=inputs[case];assert sha(ROOT/inp['path'])==inp['sha256']
   r=records[case]['result']
   if r['identity']['input_sha256']!=inp['sha256']:
    eq=equivalents[g['key'],case];assert sha(ROOT/eq['original_path'])==r['identity']['input_sha256']==eq['original_sha256']
    assert eq['packaged_input_sha256']==inp['sha256']
    with np.load(ROOT/eq['original_path']) as a,np.load(ROOT/inp['path']) as b:
     for field in eq['identical_simulation_arrays']:np.testing.assert_array_equal(a[field],b[field])
   assert r['status']=='complete' and r['final_spikes']==0 and r['final_max_mV']<0
   ans=np.array(r['ans'],int);assert len(ans)==r['length']>0
   assert np.all((ans>=0)&(ans//31<len(distances)))
   onsets=100+10*(ans%31);dist=distances[ans//31]
   counts=np.histogram(onsets,bins=np.linspace(100,400,31))[0]
   model='ca1_20212' if g['model']=='ca1' else g['model']
   expected=metrics.loc[(model,g['regime'],case)]
   np.testing.assert_allclose([len(ans),dist.mean(),gini(counts)],[expected.n_increments,expected.mean_distance_um,expected.temporal_concentration_gini30],rtol=1e-12,atol=1e-10)
   rows.append(dict(group=g['key'],case=case,length=len(ans),mean_distance_um=dist.mean(),gini30=gini(counts),input_sha256=inp['sha256']))
 for path in (ROOT/'models').glob('*/source_manifest.json'):
  for name,digest in read(path).items():assert sha(path.parent/name)==digest
 manifest=read(D/'profiles/summary_manifest.json');assert sha(D/'provenance/publication_snapshot.json')==manifest['snapshot_sha256']
 assert len(manifest['profiles'])==800
 for q in manifest['profiles']:assert sha(D/'profiles'/q['path'])==q['sha256']
 pulse=pd.DataFrame(rows);pulse.to_csv(OUT/'pulse_case_metrics.csv',index=False)
 timing=pd.read_csv(D/'figures/figure3/timing_statistics_all_completed.csv')
 raw=[]
 for pattern in ['ampa','nmda','bhigh']:
  a=timing[timing.pattern==pattern].gini_30;b=timing[timing.pattern=='sync'].gini_30
  u,p=mannwhitneyu(a,b,alternative='two-sided',method='asymptotic',use_continuity=True)
  raw.append(dict(comparison=pattern+' vs sync',U=float(u),p_raw=float(p)))
 order=np.argsort([r['p_raw'] for r in raw]);last=0
 for rank,j in enumerate(order):last=max(last,min(1,(3-rank)*raw[j]['p_raw']));raw[j]['p_holm']=last
 pd.DataFrame(raw).to_csv(OUT/'figure3_gini_tests.csv',index=False)
 voltage=pd.read_csv(D/'figures/supplementary1/autonomous_800_case_metrics.csv');assert len(voltage)==800
 passing=(voltage.max_abs_mV<=1e-3)&(voltage.rmse_mV<=1e-4)
 benchmark=pd.read_csv(D/'figures/supplementary2/benchmark.csv');assert len(benchmark)==234
 gt=list((D/'simple/groundtruth').glob('case_*/result.json'));assert len(gt)==298
 ids=set();cert=read(D/'simple/certificate_audit.json');assert cert['status']=='passed' and len(cert['cases'])==298
 for p in gt:
  r=read(p);case=r['case_id'];assert case not in ids;ids.add(case)
  greedy=read(D/'simple/ready'/f'case_{case:03d}.json')
  assert len(r['witness'])==r['minimum_length']<=greedy['ans_length'] and r['cpu_witness_peak_mV']<0
 assert ids==set(range(300))-{138,164}
 archive=read(D/'provenance/simple_cpu_certificates.json')
 assert not archive['missing'] and len(archive['references'])==298
 for f in archive['files']:
  assert sha(D/'simple/cpu_archive'/f['path'])==f['sha256']
 for q in archive['references']:
  r=read(D/'simple/groundtruth'/f"case_{q['case']:03d}"/'result.json')
  assert q['minimum']==r['minimum_length'] and q['witness']==r['witness']
  assert q['cpu_peak_mV']==r['cpu_witness_peak_mV']
  assert (D/'simple/cpu_archive'/q['certificate']).exists()
  if q['case'] in [0,26]:assert q['certificate'].startswith('validation/')
 # A numerical rerun of Fig. 6 is compared when present; no simulation is implied.
 checked=[]
 for p in (ROOT/'outputs/analysis/transfer_drive/source_data').glob('*.csv'):
  ref=D/'figures/figure6'/p.name
  if not ref.exists():continue
  a=pd.read_csv(p);b=pd.read_csv(ref);assert a.shape==b.shape and a.columns.equals(b.columns)
  for c in a:
   if pd.api.types.is_numeric_dtype(a[c]):np.testing.assert_allclose(a[c],b[c],rtol=1e-10,atol=1e-12,equal_nan=True)
   else:assert a[c].fillna('').equals(b[c].fillna(''))
  checked.append(p.name)
 result=dict(passed=True,pulse_cases=len(rows),input_hashes_checked=800,profiles_checked=800,models_checked=6,voltage_cases=len(voltage),voltage_within_thresholds=int(passing.sum()),voltage_max_error_mV=float(voltage.max_abs_mV.max()),simple_algorithm_cases=len(list((D/'simple/ready').glob('case_*.json'))),simple_minimum_cases=len(gt),simple_spiking_comparable_cases=len(benchmark),simple_exact_count=int((benchmark.difference==0).sum()),fig6_tables_recomputed_and_matched=checked,environment=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,platform=platform.platform()),scope='CPU checks of retained scientific data and statistical calculations; no new neuron simulation')
 (OUT/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
