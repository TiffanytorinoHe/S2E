"""Recompute one SIMPLE case from its fixed input row; never overwrite source data."""
from pathlib import Path
import argparse,json,os,subprocess,numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
p=argparse.ArgumentParser();p.add_argument('--case',type=int,required=True);p.add_argument('--mode',choices=['solve','verify','enumerate'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--batch',type=int,default=131072);a=p.parse_args()
assert 0<=a.case<300
out=a.output.resolve();assert out.is_relative_to(ROOT/'outputs'),'Write new runs under outputs/'
out.mkdir(parents=True,exist_ok=True);os.chdir(HERE)
if a.mode=='solve':
 subprocess.run([str(HERE/'x86_64/special'),'-nobanner','-nogui','-python',str(HERE/'opt_inh_all.py'),'--mode','solve','--setting','all_hh','--hh-table-off','--run-id',str(a.case),'--max-increments','100','--output-root',str(out)],check=True)
elif a.mode=='verify':
 result=json.loads((ROOT/'data/simple/groundtruth'/f'case_{a.case:03d}'/'result.json').read_text())
 counts=np.bincount(result['witness'],minlength=20)[None,:];np.save(out/'counts.npy',counts)
 subprocess.run(['bash',str(HERE/'cpu_verify_counts.sh'),str(a.case),str(out/'counts.npy'),str(out/'reference')],check=True)
 z=np.load(out/'reference/all_hh'/f'case_{a.case:03d}'/'rec/forward_counts.npz')
 assert np.array_equal(counts,z['counts']) and not bool(z['hh_table']) and np.max(z['voltage'])<0
 print('Independent CPU witness verified')
else:
 assert a.case not in [138,164],'These length-18 cases are outside the published enumeration'
 from groundtruth_case import run
 run(a.case,out,a.batch,0,max_length=14)
