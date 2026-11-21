"""Recompute Fig. 5 means and bootstrap contrasts from all 800 solutions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from verify_release import gini

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/analysis/pulse';OUT.mkdir(parents=True,exist_ok=True)
MODELS=['l5pc','ca1_20212','l23','msn'];PATTERNS=['ampa','nmda','bhigh','sync']
rng=np.random.default_rng(20261006);rows=[];draws={};summaries={};contrasts=[];cases=[]
for model in MODELS:
 for pattern in PATTERNS:
  d=json.loads((ROOT/f'data/solutions/pulse/{model}_{pattern}.json').read_text())
  distance=np.asarray(d['geometry']['distance_um']);maximum=distance[:d['geometry']['nonaxon_segments']].max()
  values={k:[] for k in ['mean_normalized_distance','temporal_concentration_gini30']}
  records={r['case']:r['result'] for r in d['results']}
  for case in d['selected']:
   r=records[case];a=np.array(r['ans']);assert len(a)==r['length'] and r['final_max_mV']<0
   onset=100+10*(a%31);v=distance[a//31].mean()/maximum;g=gini(np.histogram(onset,bins=np.linspace(100,400,31))[0])
   values['mean_normalized_distance'].append(v);values['temporal_concentration_gini30'].append(g)
   cases.append(dict(model=model,pattern=pattern,case=case,length=len(a),mean_normalized_distance=v,temporal_concentration_gini30=g))
  for metric,x in values.items():
   x=np.array(x);b=x[rng.integers(0,len(x),(12000,len(x)))].mean(axis=1);lo,hi=np.percentile(b,[2.5,97.5])
   draws[model,pattern,metric]=b;summaries[model,pattern,metric]=x.mean()
   rows.append(dict(model=model,pattern=pattern,metric=metric,n=len(x),mean=x.mean(),ci_low=lo,ci_high=hi))
 for key,metric,pos,neg in [('outward','mean_normalized_distance','bhigh','ampa'),('return','mean_normalized_distance','sync','bhigh'),('concentration','temporal_concentration_gini30','sync','bhigh')]:
  lo,hi=np.percentile(draws[model,pos,metric]-draws[model,neg,metric],[2.5,97.5])
  contrasts.append(dict(model=model,contrast=key,estimate=summaries[model,pos,metric]-summaries[model,neg,metric],ci_low=lo,ci_high=hi))
for name,items in [('condition_summary',rows),('contrasts',contrasts)]:
 frame=pd.DataFrame(items);reference=pd.read_csv(ROOT/f'data/figures/figure5/{name}.csv')
 assert frame.columns.equals(reference.columns) and frame.shape==reference.shape
 for c in frame:
  if pd.api.types.is_numeric_dtype(frame[c]):np.testing.assert_allclose(frame[c],reference[c],rtol=1e-10,atol=1e-12)
  else:assert frame[c].equals(reference[c])
 frame.to_csv(OUT/f'{name}.csv',index=False)
pd.DataFrame(cases).to_csv(OUT/'case_metrics.csv',index=False)
print('All 800 solutions and Fig. 5 bootstrap tables reproduced.')
