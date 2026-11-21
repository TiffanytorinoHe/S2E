"""Recompute state-conditioned step-zero transfer/drive profiles."""
import argparse,sys,subprocess
from pathlib import Path
from common import ROOT,group,identity,setup,read
def main():
 p=argparse.ArgumentParser();p.add_argument('--group',required=True);p.add_argument('--case',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 g=group(a.group);out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
 if not (out/'gate.json').exists():
  subprocess.run([sys.executable,str(ROOT/'experiments/solve.py'),'--group',a.group,'--case',str(a.case),'--output',str(out),'--gate-only'],check=True)
 gate=read(out/'gate.json');assert gate['identity']==identity(g,a.case) and gate['passed']
 m,model=setup(g,a.case,out)
 from tangent import collect
 sys.path.insert(0,str(ROOT/'src/profiles'))
 from kernel_profiles import make_profiles
 print(make_profiles(m,collect(m),out,identity(g,a.case)))
if __name__=='__main__':main()
