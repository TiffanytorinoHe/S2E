"""Autonomous S2E nonlinear forward equations, all state in plain NumPy arrays.

NEURON is used in __init__/snapshot only to copy initial model parameters/state.
step() and its callees do not access the NEURON interpreter or reference states.
"""
import re
from pathlib import Path
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
import l5_rates,ca1_rates,msn_rates

CA1_GATES={'hha_old':'mhsn','hha2':'mhsn','h':'n','nap_poirazi':'n','kad_poirazi':'nl','kap_poirazi':'nl','calH':'mh','car':'mh','somacar':'mh','km_zach':'n','mykca':'o','kca':'m','cat':'mh','cal':'m'}
def clean(s):return re.sub(r':[^\n]*','',re.sub(r'COMMENT.*?ENDCOMMENT','',s,flags=re.S))
def uniform(a):
    a=np.asarray(a,float)
    return float(a[0]) if len(a) and np.all(a==a[0]) else a

class Engine:
    def __init__(self,m,source,kind):
        from neuron import h
        import data_c2py_v2 as batch
        from ChannelType import mech_type
        from myfunc_torch import CalcMatrix
        self.kind=kind;self.n=m.Nseg;self.dt=float(h.dt);self.temp=float(h.celsius)
        matrix=CalcMatrix(m.seclist,[{},{},{},{}]).numpy();self.nn=len(matrix);self.base=csc_matrix(matrix);self.diagonal=np.diag(matrix).copy()
        self.D1=np.array([self.dt*1e5/s.cm/s.area() for s in m.seglist]);self.D2=np.array([self.dt*1e3/s.cm for s in m.seglist])
        self.gp=np.array([s.pas.g for s in m.seglist]);self.ep=np.array([s.pas.e for s in m.seglist])
        mods={}
        for f in (source/'mod').glob('*.mod'):
            text=clean(f.read_text());match=re.search(r'\bSUFFIX\s+(\w+)',text)
            if match:mods[match[1]]=(f,text)
        self.channels={};self.pools={};self.fields={};self.cpp={}
        for name in m.mech_list:
            idx=m.rec[name]['idx'].numpy().astype(int)
            if not len(idx):continue
            f,text=mods[name];cpp=(source/'x86_64'/(f.stem+'.cpp')).read_text();self.cpp[name]=cpp
            self.fields[name]={k:int(v) for k,v in re.findall(r'^#define (\w+)_columnindex (\d+)$',cpp,re.M)}
            objs=[getattr(m.seglist[i],name) for i in idx];params={}
            block=re.search(r'PARAMETER\s*{(.*?)}',text,re.S)
            for key in re.findall(r'^\s*(\w+)\b',block[1] if block else '',re.M):
                if key in ['v','dt','celsius','cai','cao','cali','calo']:continue
                if hasattr(objs[0],key):params[key]=uniform([getattr(o,key) for o in objs])
                elif hasattr(h,key+'_'+name):params[key]=float(getattr(h,key+'_'+name))
            for key in ['FARADAY','R']:
                match=re.search(r'\b'+key+r'\s*=\s*([\d.eE+-]+)\s*;',cpp)
                if match:params[key]=float(match[1])
                unit=re.search(r'_nrnunit_'+key+r'\[2\]\s*=\s*{([^,]+),([^}]+)}',cpp)
                if unit:
                    legacy=int(h.nrnunit_use_legacy()) if hasattr(h,'nrnunit_use_legacy') else 0
                    value=unit[1+legacy].strip();params[key]=float.fromhex(value) if '0x' in value else float(value)
            for key in ['tadj']:
                if hasattr(objs[0],key):params[key]=uniform([getattr(o,key) for o in objs])
                elif hasattr(h,key+'_'+name):params[key]=float(getattr(h,key+'_'+name))
                elif key in self.fields[name]:params[key]=uniform(np.asarray(batch.columns(name,np.array([self.fields[name][key]],np.int32)))[:,0])
            params['usetable']=float(getattr(h,'usetable_'+name)) if hasattr(h,'usetable_'+name) else 0
            tp,ion,ns=mech_type[name][:3]
            if name in ['CaDynamics_E2','cad','cadyn','caldyn']:
                self.pools[name]=dict(idx=idx,p=params)
                if name=='caldyn':
                    body=text[text.index('PROCEDURE state'):]
                    assert not re.search(r'\bcali\s*=',body),'Do not silently freeze an active caldyn'
                continue
            gates=CA1_GATES[name] if kind=='ca1_20212' else ('o' if kind=='msn' and tp in ['kca','sk'] else 'z' if name=='SK_E2' else 'mhn' if name=='hh_grad' else 'mh'[:ns])
            p=dict(idx=idx,p=params,ion=ion,gates=gates,type=tp)
            if kind in ['l5pc','l23']:
                if name=='hh_grad':
                    for key in ['ena','ek']:params[key]=uniform([getattr(m.seglist[i],key) for i in idx])
                elif ion!='ca':params['reversal']=uniform([o.ehcn if ion=='hcn' else getattr(m.seglist[i],'e'+ion) for o,i in zip(objs,idx)])
            elif kind=='ca1_20212':
                needed=['ena','ek'] if name.startswith('hha') else ['eh'] if name=='h' else ['ena'] if name=='nap_poirazi' else ['eca'] if name=='car' else ['ek'] if ion=='k' else []
                for key in needed:
                    if hasattr(objs[0],key):params[key]=uniform([getattr(o,key) for o in objs])
                    elif hasattr(m.seglist[idx[0]],key):params[key]=uniform([getattr(m.seglist[i],key) for i in idx])
            else:
                if 'factor' in params:assert np.all(np.asarray(params['factor'])==0),'Active PKA modulation requires independent cascade dynamics'
                if ion in ['na','k']:params['reversal']=uniform([getattr(m.seglist[i],'e'+ion) for i in idx])
            self.channels[name]=p
        self.eca0=np.zeros(self.n);self.theta=np.zeros(self.n);self.cao=np.ones(self.n);self.calo=np.ones(self.n)
        for i,s in enumerate(m.seglist):
            if hasattr(s,'cai'):
                self.cao[i]=s.cao;self.eca0[i]=s.eca
                if (int(h.ion_style('ca_ion',sec=s.sec))//64)%2:self.theta[i]=s.eca/np.log(s.cao/s.cai)
            if hasattr(s,'cali'):self.calo[i]=s.calo
        self.m=m;self.batch=batch

    def snapshot(self):
        # Reference/initialization only. Never called by step().
        m=self.m
        ca=np.array([s.cai if hasattr(s,'cai') else 1 for s in m.seglist])
        cal=np.array([s.cali if hasattr(s,'cali') else 1 for s in m.seglist])
        gs={name:np.asarray(self.batch.columns(name,np.array([self.fields[name][g] for g in sp['gates']],np.int32))).T.copy() for name,sp in self.channels.items()}
        return np.array([s.v for s in m.seglist]),ca,cal,gs

    def eca(self,ca):
        ec=self.eca0.copy();d=self.theta!=0;ec[d]=self.theta[d]*np.log(self.cao[d]/ca[d]);return ec

    def currents(self,v,ca,cal,gs):
        n=self.n;g=np.zeros(n);ge=np.zeros(n);ghki=np.zeros(n);ghkj=np.zeros(n);ica=np.zeros(n);ec=self.eca(ca)
        for name,s in self.channels.items():
            idx=s['idx'];p=s['p'];x=gs[name];ion=s['ion'];vv=v[idx];c=ca[idx]
            if self.kind in ['l5pc','l23']:
                if name=='hh_grad':
                    gn=p['gnabar']*x[0]**3*x[1];gk=p['gkbar']*x[2]**4
                    g[idx]+=gn+gk+p['gl'];ge[idx]+=gn*p['ena']+gk*p['ek']+p['gl']*p['el'];continue
                power=2 if name in ['Ca_HVA','Ca_LVAst','K_Pst'] else 3 if name in ['NaTa_t','NaTs2_t','NaTg','Nap_Et2'] else 4 if name=='K_Tst' else 1
                gc=p['g'+name+'bar']*x[0]**power
                if len(x)==2:gc*=x[1]
                rev=ec[idx] if ion=='ca' else p['reversal']
            elif self.kind=='ca1_20212':
                if name.startswith('hha'):
                    gn=p['gnabar']*x[0]**2*x[1]*x[2];gk=p['gkbar']*x[3]**2
                    g[idx]+=gn+gk+p['gl'];ge[idx]+=gn*p['ena']+gk*p['ek']+p['gl']*p['el'];continue
                if name in ['cal','cat']:
                    pref=p['gcalbar']*x[0] if name=='cal' else p['gcatbar']*x[0]**2*x[1]
                    pref*=p['ki']/(p['ki']+c)
                    def ghk(zv):
                        f=(25/293.15)*(self.temp+273.15)/2;z=zv/f
                        with np.errstate(invalid='ignore',divide='ignore'):b=np.where(abs(z)<1e-4,1-z/2,z/(np.exp(z)-1))
                        return -f*(1-c/self.cao[idx]*np.exp(z))*b
                    cur=pref*ghk(vv);j=pref*(ghk(vv+.001)-ghk(vv))/.001
                    ghki[idx]+=cur;ghkj[idx]+=j
                    if name=='cal':ica[idx]+=cur
                    continue
                if name=='h':gc=p['gbar']*x[0];rev=p['eh']
                elif name=='nap_poirazi':gc=p['gnabar']*x[0]**3;rev=p['ena']
                elif name in ['kad_poirazi','kap_poirazi']:gc=p['gkabar']*x[0]*x[1];rev=p['ek']
                elif name in ['calH','car','somacar']:
                    gc=p['gcalbar' if name=='calH' else 'gcabar']*x[0]**3*x[1];rev=p['eca'] if name=='car' else ec[idx]
                elif name=='km_zach':gc=1e-4*p['tadj']*p['gbar']*x[0];rev=p['ek']
                elif name=='mykca':gc=p['gkbar']*x[0]**p['st'];rev=p['ek']
                elif name=='kca':gc=p['gbar']*x[0]**3;rev=p['ek']
                else:raise ValueError(name)
            else:
                if s['type']=='ghk':
                    power=1 if name=='cal12' else 2 if name in ['cal13','can_kai','caq_kai'] else 3
                    pref=p['pbar']*x[0]**power
                    if len(x)==2:pref*=x[1]*p['a']+1-p['a'] if name in ['cal12','can_kai'] else x[1]
                    ci,co=(cal[idx],self.calo[idx]) if ion=='cal' else (c,self.cao[idx])
                    def ghk(zv):
                        z=.001*2*p['FARADAY']*zv/(p['R']*(self.temp+273.15));z=np.where(z==0,z+1e-6,z)
                        return .001*2*p['FARADAY']*(ci*(-z)/(np.exp(-z)-1)-co*z/(np.exp(z)-1))
                    cur=pref*ghk(vv);j=pref*(ghk(vv+.001)-ghk(vv))/.001
                    ghki[idx]+=cur;ghkj[idx]+=j
                    if ion=='ca':ica[idx]+=cur
                    continue
                power=3 if name=='naf_kai' else 2 if name in ['kaf_kai','kas_kai'] else 1
                gc=p['gbar']*x[0]**power
                if len(x)==2:gc*=x[1]
                rev=p['reversal']
            g[idx]+=gc;ge[idx]+=gc*rev
            if ion=='ca':ica[idx]+=gc*(vv-rev)
        return g,ge,ghki,ghkj,ica

    def update_gates(self,v,ca,gs):
        for name,s in self.channels.items():
            idx=s['idx'];p=s['p'];vv=v[idx];cc=ca[idx]
            if self.kind=='ca1_20212':rates=ca1_rates.rates(name,vv,cc,p,self.temp,self.dt)
            elif self.kind=='msn':rates=msn_rates.rates(name,vv,cc,p,self.temp,self.dt)
            elif name=='NaTg':
                v1=np.where(vv==-38,vv+.0001,vv);a=.182*(v1+38-p['vshiftm'])/(1-np.exp(-(v1+38-p['vshiftm'])/p['slopem']));b=.124*(-v1-38+p['vshiftm'])/(1-np.exp(-(-v1-38+p['vshiftm'])/p['slopem']))
                mi=a/(a+b);mt=1/(a+b)/l5_rates.QT;v1=np.where(v1==-66,v1+.0001,v1)
                a=-.015*(v1+66-p['vshifth'])/(1-np.exp((v1+66-p['vshifth'])/p['slopeh']));b=-.015*(-v1-66+p['vshifth'])/(1-np.exp((-v1-66+p['vshifth'])/p['slopeh']))
                rates=[(mi,mt),(a/(a+b),1/(a+b)/l5_rates.QT)]
            else:rates=l5_rates.rates(name,vv,cc,self.temp)
            for k,(inf,tau) in enumerate(rates):
                if isinstance(tau,tuple):factor=tau[1]
                elif np.isscalar(tau) and tau==0:factor=1.
                else:factor=1-np.exp(-self.dt/tau)
                gs[name][k]+=factor*(inf-gs[name][k])

    def step(self,v,ca,cal,gs,w):
        g,ge,fi,fj,ica=self.currents(v,ca,cal,gs)
        ga,gn,gi=w;C=.267;rho=.062
        q=1/(1+C*np.exp(-rho*v));q1=1/(1+C*np.exp(-rho*(v+.001)))
        inward=-ga*v-gn*q*v+gi*(-70-v)
        j=(-ga*.001-gn*(q1*(v+.001)-q*v)-gi*.001)/.001
        rhs=np.zeros(self.nn);rhs[:self.n]=v+self.D2*(self.gp*self.ep+ge-fi+fj*v)+self.D1*(inward-j*v)
        a=self.base.copy();d=self.diagonal.copy();d[:self.n]+=self.D2*(g+fj)-self.D1*j;a.setdiag(d)
        vn=spsolve(a,rhs,permc_spec='NATURAL')[:self.n]
        ca=ca.copy()
        for name,s in self.pools.items():
            if name=='caldyn':continue
            ix=s['idx'];p=s['p'];F=p['FARADAY']
            if name=='CaDynamics_E2':inf=p['minCai']-10000*ica[ix]*p['gamma']/(2*F*p['depth'])*p['decay'];tau=p['decay']
            elif name=='cad':
                # Native cad.mod uses taur/7 for relaxation, but taur/18/7 in cInf.
                inf=np.maximum(-10000*ica[ix]/(2*F*p['depth']),0)*p['taur']/18/7+p['cainf'];tau=p['taur']/7
            elif name=='cadyn':
                tau=p['taur'];drive=np.maximum(-p['drive']*ica[ix]/(2*F*p['depth']),0);pump=-p['kt']*ca[ix]/(ca[ix]+p['kd'])
                inf=(drive+p['pump']*pump)*tau+p['cainf']
            ca[ix]+=(1-np.exp(-self.dt/tau))*(inf-ca[ix])
        self.update_gates(vn,ca,gs)
        return vn,ca,cal,gs
