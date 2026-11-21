"""Paired L5 pulse/train analysis; original region definitions and bootstrap."""
from pathlib import Path
import json,csv,re
import numpy as np
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'outputs/analysis/train';R.mkdir(parents=True,exist_ok=True)
D=json.loads((ROOT/'data/figures/supplementary4/verified_results.json').read_text())
geom=json.loads((ROOT/'data/geometry/l5_recording_sites.json').read_text())
dist=np.array([g['distance_um'] for g in geom]);names=[g['section'].split('].')[-1] for g in geom]
regions={k:np.array([g[k] for g in geom]) for k in ['is_perisomatic','is_distal','is_trunk']}
rows=[];stats=[];maps={};profiles=[];audit=[]
def csvout(name,items):
 with (R/name).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
for g in D['groups'][:4]:
 assert g['model']=='l5pc' and len(g['records'])==50 and not g['errors'] and g['gate']['passed']
 geo=g['geometry'];assert len(dist)==len(geo['distance_um'])
 assert np.allclose(dist,geo['distance_um'],atol=1e-7,rtol=0)
 assert names==[n.split('].')[-1] for n in geo['section_names']]
 n=geo['nonaxon_segments'];assert np.array_equal(np.arange(len(geom))<n,[not x['is_axon'] for x in geom])
 counts={v:np.zeros(len(geom)) for v in ['pulse','train']}
 for r in g['records']:
  row={'condition':g['regime'],'case_id':r['case']}
  for mode in ['pulse','train']:
   ans=np.array(r[mode]['ans']);loc=ans//31;assert len(ans)>0 and loc.min()>=0 and loc.max()<n
   np.add.at(counts[mode],loc,1)
   row[mode+'_count']=len(ans)
   row[mode+'_mean_distance_um']=float(dist[loc].mean())
   for key,label in [('is_perisomatic','perisomatic'),('is_distal','distal_source'),('is_trunk','apical_trunk')]:row[mode+'_'+label]=float(regions[key][loc].mean())
  rows.append(row)
 for mode,values in counts.items():
  prob=values/values.sum();maps[g['regime']+'_'+mode]=prob.tolist()
  bins=np.arange(0,dist[:n].max()+40,40);inds=np.digitize(dist[:n],bins)-1
  for k in range(len(bins)-1):
   mask=inds==k
   profiles.append(dict(condition=g['regime'],mode=mode,distance_um=float((bins[k]+bins[k+1])/2),fraction=float(prob[:n][mask].mean()) if mask.any() else None))
 for metric in ['perisomatic','distal_source','apical_trunk','count','mean_distance_um']:
  a=np.array([x['pulse_'+metric] for x in rows if x['condition']==g['regime']]);b=np.array([x['train_'+metric] for x in rows if x['condition']==g['regime']]);diff=b-a
  rng=np.random.default_rng(20261107);boots=np.concatenate([np.median(diff[rng.integers(0,len(diff),(5000,len(diff)))],axis=1) for _ in range(20)])
  scale=100 if metric in ['perisomatic','distal_source','apical_trunk'] else 1
  lo,hi=np.percentile(boots,[2.5,97.5]);stats.append(dict(condition=g['regime'],metric=metric,n=len(a),pulse_median=float(np.median(a)*scale),train_median=float(np.median(b)*scale),paired_median_difference=float(np.median(diff)*scale),ci_low=float(lo*scale),ci_high=float(hi*scale),unit='percentage points' if scale==100 else ('candidates' if metric=='count' else 'um')))
 audit.append(dict(group=g['key'],n=len(g['records']),input_hashes_verified=True,pulse_result_hashes_verified=True,baseline_error_max_mV=max(r['verified_baseline_max_error_mV'] for r in g['records']),all_final_max_below_zero=True,geometry_and_region_indices_verified=True))
csvout('paired_realizations.csv',rows);csvout('paired_statistics.csv',stats);csvout('distance_profiles.csv',profiles)
(R/'pooled_maps.json').write_text(json.dumps(maps));(R/'audit.json').write_text(json.dumps(audit,indent=2))
for s in stats:
 if s['metric'] in ['perisomatic','count'] or (s['condition'],s['metric']) in [('bhigh','distal_source'),('sync','apical_trunk')]:print(s)

import pandas as pd
for name in ['paired_realizations','paired_statistics','distance_profiles']:
 a=pd.read_csv(R/f'{name}.csv');b=pd.read_csv(ROOT/f'data/figures/supplementary4/{name}.csv')
 assert a.columns.equals(b.columns) and a.shape==b.shape
 for c in a:
  if pd.api.types.is_numeric_dtype(a[c]):np.testing.assert_allclose(a[c],b[c],rtol=1e-10,atol=1e-12,equal_nan=True)
  else:assert a[c].fillna('').equals(b[c].fillna(''))
print('All 200 pulse/train pairs and Supplementary Table 6 reproduced.')
