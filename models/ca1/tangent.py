"""Discrete tangent for the actual CA1_20212 NEURON forward implementation.

Four-state Na/K, normalized GHK, dummy-ion T current, one active Ca pool.
Uses named compiled fields, never another model's positional channel layout.
"""
from pathlib import Path
import re,time
import numpy as np
from neuron import h
import data_c2py_v2 as batch
from myfunc_torch import CalcMatrix
from model import initialize
from tangent_base import tree_structure,cpu_reference,gpu

# Mechanism: gate names, exported conductance, reversal source.
SPEC={
 'hha_old':('mhsn',None,None),'hha2':('mhsn',None,None),
 'h':('n','g','eh'),'nap_poirazi':('n','gna','ena'),
 'kad_poirazi':('nl','gka','ek'),'kap_poirazi':('nl','gka','ek'),
 'calH':('mh','gcal','eca'),'car':('mh','gca','eca_fixed'),
 'somacar':('mh','gca','eca'),'km_zach':('n','gk','ek'),
 'mykca':('o','gk','ek'),'kca':('m','gk','ek'),
 'cat':('mh','gcat',None),'cal':('m','gcal',None)}

def ghk_parts(v,ci,co,temperature,ki):
 f=(25./293.15)*(temperature+273.15)/2;z=v/f;e=np.exp(z);small=np.abs(z)<1e-4
 zz=np.where(small,1.,z);den=np.expm1(zz)
 b=np.where(small,1-z/2,zz/den)
 bp=np.where(small,-.5,(den-zz*np.exp(zz))/den**2)
 F=f*((ci/co)*e-1)*b
 Fv=(ci/co)*e*b+((ci/co)*e-1)*bp
 Fc=f/co*e*b
 h2=ki/(ki+ci)
 return h2*F,h2*Fv,h2*(Fc-F/(ki+ci))

def collect(m):
 tic=time.perf_counter();initialize(m)
 matrix=CalcMatrix(m.seclist,[{},{},{},{}]).numpy();n=m.Nseg;nn=len(matrix);T=round(m.meta['stop']/h.dt);soma=m.seg2idx[m.soma]
 parent,order,lo,up=tree_structure(matrix,soma)
 D1=np.array([h.dt*1e5/s.cm/s.area() for s in m.seglist]);D2=np.array([h.dt*1e3/s.cm for s in m.seglist])
 root=Path(__file__).resolve().parent;cpp={};fields={}
 for p in (root/'mod').glob('*.mod'):
  match=re.search(r'\bSUFFIX\s+(\w+)',p.read_text())
  if match:cpp[match[1]]=root/'x86_64'/(p.stem+'.cpp')
 for name,path in cpp.items():fields[name]={k:int(v) for k,v in re.findall(r'^#define (\w+)_columnindex (\d+)$',path.read_text(),re.M)}
 def columns(name,names):return np.array([fields[name][k] for k in names],np.int32)
 idx={name:m.hines2idx[batch.get_ml_nodeindices(name)] for name in m.mech_list}
 specs={};segments=[]
 for name,(gates,g,rev) in SPEC.items():
  ind=idx[name];slots=[]
  for gate in gates:slots.append(np.arange(len(segments),len(segments)+len(ind)));segments.extend(ind.tolist())
  if name in ['hha_old','hha2']:names=['gna','gk','gl']+['dgd'+q for q in gates]
  elif name in ['cat','cal']:names=['iCa' if name=='cat' else 'ica',g]+['did'+q for q in gates]
  else:names=[g]+['dgd'+q for q in gates]
  names+=['d'+q+'d'+q+'_' for q in gates]+['d'+q+'dv' for q in gates]
  if name in ['mykca','kca']:names+=['d'+gates+'dc']
  if rev in ['eh','eca_fixed']:names+=['eh' if rev=='eh' else 'eca']
  specs[name]={'idx':ind,'gates':gates,'conductance':g,'rev':rev,'slots':slots,'names':names,'cols':columns(name,names)}
 segments=np.array(segments,np.int32);K=len(segments)
 vcoef=np.zeros((T,4,n));gcoef=np.zeros((T,5,K));diag=np.zeros((T,nn));vnext=np.zeros((T,n));wave=np.zeros((T,m.N_i));trace=np.zeros(T+1);trace[0]=m.soma.v
 cad=idx['cad'];cadcols=columns('cad',['dcdi','dcdc_']);theta=np.zeros(n);cai=np.ones(n);cao=np.ones(n);ena=np.zeros(n);ek=np.zeros(n);eca=np.zeros(n)
 for i,s in enumerate(m.seglist):
  if hasattr(s,'ena'):ena[i]=s.ena
  if hasattr(s,'ek'):ek[i]=s.ek
  if hasattr(s,'cai'):
   cai[i]=s.cai;cao[i]=s.cao;eca[i]=s.eca
   if (int(h.ion_style('ca_ion',sec=s.sec))//64)%2:theta[i]=s.eca/np.log(s.cao/s.cai)
 exc='Exp2Syn_Excitatory_back';bg='Exp2Syn_back'
 def syncols(stem,names):
  ff={k:int(v) for k,v in re.findall(r'^#define (\w+)_columnindex (\d+)$',(root/'x86_64'/(stem+'.cpp')).read_text(),re.M)}
  return np.array([ff[k] for k in names],np.int32)
 excols=syncols('exp2syn_exc_back',['AMPA_g','NMDA_g','NMDA_C','NMDA_rho','w','AMPA_e','NMDA_e']);bgcols=syncols('exp2syn_back',['g','w'])
 exidx=m.hines2idx[batch.get_ml_nodeindices(exc)];bgidx=m.hines2idx[batch.get_ml_nodeindices(bg)]
 for name in ['cal','cat']:
  assert re.search(r'\bki\s*=\s*0?\.001\b',(root/'mod'/(name+'.mod')).read_text()),'Unexpected GHK calcium-inactivation parameter'
 for t in range(T):
  old=np.array([s.v for s in m.seglist]);cai[cad]=[m.seglist[i].cai for i in cad]
  for name in ['calH','somacar']:eca[idx[name]]=[m.seglist[i].eca for i in idx[name]]
  h.fadvance();nxt=np.array([s.v for s in m.seglist]);dv=nxt-old;vnext[t]=nxt;trace[t+1]=m.soma.v
  ja=np.zeros(n)
  for syn in m.synlist_i:ja[m.seg2idx[syn.get_segment()]]-=syn.w*syn.g
  wave[t]=[syn.g for syn in m.synlist_i]
  bd=np.asarray(batch.columns(bg,bgcols));ja+=np.bincount(bgidx,weights=-bd[:,0]*bd[:,1],minlength=n);js=ja.copy();djs=np.zeros(n)
  ga,gn,C,rho,w,ea,en=np.asarray(batch.columns(exc,excols)).T;vv=old[exidx]
  q=1/(1+C*np.exp(-rho*vv));q1=1/(1+C*np.exp(-rho*(vv+.001)))
  j0=w*(-ga+gn*rho*q*(1-q)*(en-vv)-gn*q);j1=w*(-ga+gn*rho*q1*(1-q1)*(en-vv-.001)-gn*q1)
  sec=-w*(ga*.001+gn*(q1*(vv+.001-en)-q*(vv-en)))/.001
  ja+=np.bincount(exidx,weights=j0,minlength=n);js+=np.bincount(exidx,weights=sec,minlength=n);djs+=np.bincount(exidx,weights=(j1-j0)/.001,minlength=n)
  slope=np.zeros(n);rhs_v=1+D1*(ja-js+djs*dv);rhs_c=np.zeros(n);ca_v=np.zeros(n);ca_c=np.zeros(n);ca_gate=np.zeros(K)
  for name,sp in specs.items():
   ind=sp['idx'];ar0=np.asarray(batch.columns(name,sp['cols']));ar={k:ar0[:,j] for j,k in enumerate(sp['names'])};gates=sp['gates'];slots=sp['slots']
   if name in ['cal','cat']:
    F,Fv,Fc=ghk_parts(old[ind],cai[ind],cao[ind],m.meta['temperature'],.001)
    F1,Fv1,Fc1=ghk_parts(old[ind]+.001,cai[ind],cao[ind],m.meta['temperature'],.001)
    Fs=(F1-F)/.001;Fsv=(Fv1-Fv)/.001;Fsc=(Fc1-Fc)/.001
    pref=ar['ica' if name=='cal' else 'iCa']/F
    slope[ind]+=pref*Fs;rhs_v[ind]-=D2[ind]*pref*(Fv-Fs+Fsv*dv[ind]);rhs_c[ind]-=D2[ind]*pref*(Fc+Fsc*dv[ind])
    if name=='cal':ca_v[ind]+=pref*Fv;ca_c[ind]+=pref*Fc
    for gate,sl in zip(gates,slots):
     gp=ar['did'+gate]/F;gcoef[t,0,sl]=-D2[ind]*gp*(F+Fs*dv[ind])
     if name=='cal':ca_gate[sl]=gp*F
   elif name in ['hha_old','hha2']:
    assert np.all(ar['gl']==0),'Additional HH leak must be included explicitly'
    slope[ind]+=ar['gna']+ar['gk']
    for gate,sl in zip(gates,slots):gcoef[t,0,sl]=D2[ind]*((ek[ind] if gate=='n' else ena[ind])-nxt[ind])*ar['dgd'+gate]
   else:
    g=ar[sp['conductance']];slope[ind]+=g
    rev={'ena':ena,'ek':ek,'eca':eca}.get(sp['rev'])
    rev=rev[ind] if rev is not None else ar['eh' if sp['rev']=='eh' else 'eca']
    isca=name in ['calH','car','somacar'];ep=-theta[ind]/cai[ind] if name in ['calH','somacar'] else np.zeros(len(ind))
    if isca:ca_v[ind]+=g;ca_c[ind]-=g*ep;rhs_c[ind]+=D2[ind]*g*ep
    for gate,sl in zip(gates,slots):
     dg=ar['dgd'+gate];gcoef[t,0,sl]=D2[ind]*(rev-nxt[ind])*dg
     if isca:ca_gate[sl]=(old[ind]-rev)*dg
   for gate,sl in zip(gates,slots):
    gcoef[t,2,sl]=ar['d'+gate+'d'+gate+'_'];gcoef[t,3,sl]=ar['d'+gate+'dv']
    if name in ['mykca','kca']:gcoef[t,4,sl]=ar['d'+gate+'dc']
  cadar=np.asarray(batch.columns('cad',cadcols));beta=np.zeros(n);alpha=np.zeros(n);beta[cad]=cadar[:,0];alpha[cad]=cadar[:,1]
  vcoef[t,0]=rhs_v;vcoef[t,1]=rhs_c;vcoef[t,2]=alpha+beta*ca_c;vcoef[t,3]=beta*ca_v;gcoef[t,1]=ca_gate*beta[segments]
  d=np.diag(matrix).copy();d[:n]+=D2*slope-D1*js;diag[t]=d
 factor=np.asarray(batch.factor_tree(diag,order,parent,lo,up));assert np.all(np.isfinite(diag)) and np.min(abs(diag))>1e-12
 return {'n':n,'nn':nn,'K':K,'T':T,'soma':soma,'parent':parent,'order':order,'lower':lo,'upper':up,'vcoef':vcoef,'gcoef':gcoef,'diag':diag,'factor':factor,'segments':segments,'vnext':vnext,'wave':wave,'D1':D1,'trace':trace,'collect_seconds':time.perf_counter()-tic}
