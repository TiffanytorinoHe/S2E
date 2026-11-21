"""Portable access to the publication's frozen cohorts and scientific runtime."""
from pathlib import Path
import hashlib,json,os,sys,shutil
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2),encoding='utf-8');t.replace(p)
def group(key):return next(g for g in read(ROOT/'data/cohorts.json')['groups'] if g['key']==key)
def identity(g,c,variant='pulse'):
 i=next(i for i in g['inputs'] if i['case']==c)
 return dict(group=g['key'],case=c,variant=variant,input_sha256=i['sha256'],source_release_sha256=g['source_release_sha256'],runtime_manifest_sha256=sha(ROOT/'models'/g['runtime']/'source_manifest.json'))
def setup(g,c,d,variant='pulse'):
 if c not in g['selected']:raise ValueError('Case is not in the frozen publication cohort')
 src=ROOT/'models'/g['runtime'];manifest=read(src/'source_manifest.json')
 for name,digest in manifest.items():
  if sha(src/name)!=digest:raise ValueError('Scientific source mismatch: '+name)
 item=next(i for i in g['inputs'] if i['case']==c);inp=ROOT/item['path']
 if sha(inp)!=item['sha256']:raise ValueError('Input identity mismatch')
 if 'model' in sys.modules:raise RuntimeError('Use one fresh process per case')
 runtime=Path(d).resolve()/'runtime';(runtime/'input').mkdir(parents=True,exist_ok=True)
 link=runtime/'input'/f"{g['regime']}_{c}.npz"
 if not link.exists():shutil.copyfile(inp,link)
 assert sha(link)==item['sha256']
 os.chdir(src);sys.path.insert(0,str(src));sys.path.insert(0,str(ROOT/'environment/build/bridges'))
 import model
 model.HERE=runtime
 if g['new_msn_config']:os.environ['CAL_INPUT']=str(inp)
 m=model.build(g['regime'],c,variant=variant)
 if not len(m.meta['old_v']):m.meta['stop']=400.025
 from neuron import h
 assert float(h.dt)==.025 and abs(m.meta['stop']-400.025)<1e-8
 for flag in ['usetable_hh','usetable_hh_grad']:
  if hasattr(h,flag):assert float(getattr(h,flag))==0
 save(Path(d)/'settings.json',dict(dt_ms=float(h.dt),celsius=float(h.celsius),v_init_mV=float(m.v_r),stop_ms=m.meta['stop'],segments=[dict(name=s.name(),length_um=float(s.L),nseg=int(s.nseg)) for s in m.seclist],counts=list(m.N_back),identity=identity(g,c,variant)))
 return m,model
