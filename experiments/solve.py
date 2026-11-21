"""Frozen S2E greedy search; one process owns one GPU and one realization.

The validated model/tangent implementations are used without numerical edits.
Repeated candidate selections are retained. No random inputs are generated.
"""
import argparse,os,time,traceback,subprocess,sys
from pathlib import Path
import numpy as np
from common import ROOT,group,identity,setup,read,save

def derivative_check(m,model,data,ids):
 from tangent import gpu,cpu_reference
 loc,oi=model.candidate_arrays(m);ids=np.asarray(ids,int)
 tr=gpu(data,loc[ids],oi[ids],True)['trace'];w=m.w_i.copy();rows=[]
 for j,a in enumerate(ids):
  probe=int(oi[a]);syn=m.synlist_i[probe];orig=syn.get_segment();syn.loc(m.seglist[int(loc[a])]);prev=None
  try:
   for eps in [1e-6,1e-7,1e-8,1e-9]:
    wp=w.copy();wm=w.copy();wp[probe]+=eps;wm[probe]-=eps
    fd=(model.forward(m,wp)-model.forward(m,wm))/(2*eps);norm=float(np.linalg.norm(fd))
    r=dict(candidate=int(a),epsilon=eps,relative_l2=float(np.linalg.norm(fd-tr[:,j])/max(norm,1e-30)),fd_norm=norm)
    if prev is not None:r['fd_change']=float(np.linalg.norm(fd-prev)/max(norm,1e-30))
    rows.append(r);prev=fd
  finally:
   syn.loc(orig);m.w_i[:]=w
   for s,z in zip(m.synlist_i,w):s.w=float(z)
 cpu=cpu_reference(data,loc[ids[:2]],oi[ids[:2]])['trace']
 relative=float(np.linalg.norm(cpu-tr[:,:2])/max(np.linalg.norm(cpu),1e-30))
 passed=relative<1e-10 and all(any(r['candidate']==int(a) and r['relative_l2']<1e-3 and r.get('fd_change',1)<5e-3 and r['fd_norm']>1e-8 for r in rows) for a in ids)
 return dict(passed=bool(passed),cpu_gpu_relative_l2=relative,finite_difference=rows)

def run(args):
 g=group(args.group);out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
 if args.variant=='train' and g['model']!='l5pc':raise ValueError('The published train comparison is L5 PC; do not pool other models')
 if (out/'error.json').exists():raise RuntimeError('Preserved failure: inspect before creating a new run directory')
 ident=identity(g,args.case,args.variant)
 if (out/'solve.json').exists():
  assert read(out/'solve.json')['identity']==ident
  print('Existing result retained');return
 if not args.gate_only and not (out/'gate.json').exists():
  subprocess.run([sys.executable,str(Path(__file__).resolve()),'--group',args.group,'--case',str(args.case),'--variant',args.variant,'--output',str(out),'--gate-only'],check=True)
 m,model=setup(g,args.case,out,args.variant)
 from tangent import collect,gpu
 from gpu_check import check
 check();v=model.forward(m);assert np.isfinite(v).all()
 assert np.sum((v[:-1]<0)&(v[1:]>=0))>0,'Frozen cohort must have baseline spikes'
 loc,oi=model.candidate_arrays(m)
 if args.gate_only:
  data=collect(m);assert np.array_equal(v,data['trace'])
  scores=gpu(data,loc,oi)['min'];best=int(np.argmin(scores));assert scores[best]<0
  ids=list(dict.fromkeys([5,int(np.argmax(m.distance[:m.Nseg_no_axon]))*31+10,best]))
  zero=derivative_check(m,model,data,ids)
  del data
  model.add_event(m,best)
  data=collect(m);nonzero=derivative_check(m,model,data,ids)
  gate=dict(passed=zero['passed'] and nonzero['passed'],zero=zero,nonzero=nonzero,identity=ident);save(out/'gate.json',gate)
  assert gate['passed'],gate
  del data
  return
 gate=read(out/'gate.json');assert gate['identity']==ident and gate['passed']
 np.save(out/'baseline_voltage.npy',v)
 cp=read(out/'checkpoint.json') if (out/'checkpoint.json').exists() else dict(identity=ident,ans=[],history=[])
 assert cp['identity']==ident and len(cp['ans'])==len(cp['history'])
 for a in cp['ans']:model.add_event(m,a)
 for step in range(args.steps+1):
  v=model.forward(m);assert np.isfinite(v).all()
  if v.max()<0:
   np.save(out/'final_voltage.npy',v)
   save(out/'solve.json',dict(identity=ident,status='complete',ans=cp['ans'],length=len(cp['ans']),final_spikes=0,final_max_mV=float(v.max())))
   return
  if step==args.steps:return
  tic=time.monotonic();data=collect(m);assert np.array_equal(v,data['trace'])
  scores=gpu(data,loc,oi)['min'];assert np.isfinite(scores).all();a=int(np.argmin(scores));assert scores[a]<0
  cp['history'].append(dict(iteration=len(cp['ans']),selected_id=a,score=float(scores[a]),elapsed_s=time.monotonic()-tic));cp['ans'].append(a);del data
  model.add_event(m,a);save(out/'checkpoint.json',cp);print('increments',len(cp['ans']),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--group',required=True);p.add_argument('--case',type=int,required=True);p.add_argument('--variant',choices=['pulse','train'],default='pulse');p.add_argument('--steps',type=int,default=15);p.add_argument('--gate-only',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 a.output=a.output.resolve()
 assert a.steps>0
 try:run(a)
 except Exception:
  save(a.output.resolve()/'error.json',dict(traceback=traceback.format_exc()));raise
