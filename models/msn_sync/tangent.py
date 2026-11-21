"""MSN discrete tangent for the historical FIG2_D forward model.

GHK uses NEURON's voltage secant in the matrix, with derivatives of that
secant in the tangent RHS. cadyn is differentiated at its OLD calcium.
Historical caldyn does not update cali: assert this and propagate zero cali
sensitivity rather than using its misleading nonzero derivative exports.
"""
import re,time
from pathlib import Path
import numpy as np
from neuron import h
import data_c2py_v2 as batch
from myfunc_torch import CalcMatrix
from ChannelType import mech_type
from model import initialize
from tangent_base import tree_structure,cpu_reference,gpu

def ghk_parts(v,ci,co,temperature):
    A=.001*2*96485.33212;k=A/(8.31446261815324*(273.15+temperature));z=k*v
    small=np.abs(z)<1e-4
    zs=np.where(small,1.,z);e=np.expm1(zs)
    b=zs/e;bp=(e-zs*np.exp(zs))/e**2
    b=np.where(small,1-z/2+z*z/12-z**4/720,b)
    bp=np.where(small,-.5+z/6-z**3/180,bp)
    return A*(ci*z+(ci-co)*b),A*k*(ci+(ci-co)*bp),A*(z+b)

def collect(m):
    tic=time.perf_counter();initialize(m)
    matrix=CalcMatrix(m.seclist,[{},{},{},{}]).numpy();n=m.Nseg;nn=len(matrix);T=round(m.meta['stop']/h.dt);soma=m.seg2idx[m.soma]
    parent,order,lo,up=tree_structure(matrix,soma)
    D1=np.array([h.dt*1e5/s.cm/s.area() for s in m.seglist]);D2=np.array([h.dt*1e3/s.cm for s in m.seglist])
    index={name:m.rec[name]['idx'].cpu().numpy().astype(int) for name in m.mech_list}
    specs={};segments=[];root=Path(__file__).resolve().parent
    cppmap={}
    for p in (root/'mod').glob('*.mod'):
        text=p.read_text();match=re.search(r'\bSUFFIX\s+(\w+)',text)
        if match:cppmap[match[1]]=root/'x86_64'/(p.stem+'.cpp')
    def cols_for(path,names):
        fields={k:int(v) for k,v in re.findall(r'^#define (\w+)_columnindex (\d+)$',path.read_text(),re.M)}
        return np.array([fields[k] for k in names],np.int32)
    for name in m.mech_list:
        idx=index[name];tp,ion,ns=mech_type[name][:3]
        if tp=='dyn' or not len(idx):continue
        gates=['o'] if tp in ['sk','kca'] else ['m','h'][:ns]
        slots=[]
        for gate in gates:
            slots.append(np.arange(len(segments),len(segments)+len(idx)));segments.extend(idx.tolist())
        if tp=='ghk':
            names=['i'+ion,'didv','didc']+['did'+g for g in gates]
        else:names=['gna' if ion=='na' else 'gk','e'+ion]+['dgd'+g for g in gates]
        names+=['d'+g+'d'+g+'_' for g in gates]
        if tp!='sk':names+=['d'+g+'dv' for g in gates]
        if tp in ['sk','kca']:names+=['d'+g+'dc' for g in gates]
        specs[name]={'idx':idx,'type':tp,'ion':ion,'ns':ns,'slots':slots,'names':names,'columns':cols_for(cppmap[name],names)}
    segments=np.array(segments,np.int32);K=len(segments)
    vcoef=np.zeros((T,4,n));gcoef=np.zeros((T,5,K));diag=np.zeros((T,nn));vnext=np.zeros((T,n));wave=np.zeros((T,m.N_i));trace=np.zeros(T+1);trace[0]=m.soma.v
    cadidx=index['cadyn'];calidx=index['caldyn'];frozen=np.array([m.seglist[i].cali for i in calidx])
    cai=np.ones(n);cao=np.ones(n);cali=np.ones(n);calo=np.ones(n)
    cao[cadidx]=[m.seglist[i].cao for i in cadidx];calo[calidx]=[m.seglist[i].calo for i in calidx]
    dyn=[m.seglist[i].cadyn for i in cadidx]
    taur=np.array([s.taur for s in dyn]);pump=np.array([s.pump for s in dyn])
    # kt,kd,depth are source PARAMETERs, not RANGE; verify their exact defaults.
    source=(root/'mod/cadyn.mod').read_text()
    for key,value in [('kt','1e-4'),('kd','1e-4'),('depth','0.1')]:assert re.search(r'\b'+key+r'\s*=\s*'+re.escape(value)+r'\b',source)
    drive=np.array([s.drive for s in dyn]);decay=np.exp(-h.dt/taur)
    exname='Exp2Syn_Excitatory_back';exidx=m.hines2idx[batch.get_ml_nodeindices(exname)]
    excols=cols_for(root/'x86_64/exp2syn_exc_back.cpp',['AMPA_g','NMDA_g','NMDA_C','NMDA_rho','w','AMPA_e','NMDA_e'])
    bgname='Exp2Syn_back';bgidx=m.hines2idx[batch.get_ml_nodeindices(bgname)]
    bgcols=cols_for(root/'x86_64/exp2syn_back.cpp',['g','w'])
    for t in range(T):
        old=np.array([s.v for s in m.seglist]);cai[cadidx]=[m.seglist[i].cai for i in cadidx];cali[calidx]=frozen
        h.fadvance();nxt=np.array([s.v for s in m.seglist]);dv=nxt-old;vnext[t]=nxt;trace[t+1]=m.soma.v
        assert np.array_equal(np.array([m.seglist[i].cali for i in calidx]),frozen),'caldyn forward changed; a second active calcium tangent is required'
        ja=np.zeros(n);js=np.zeros(n);djs=np.zeros(n)
        for s in m.synlist_i:
            i=m.seg2idx[s.get_segment()];ja[i]-=s.w*s.g
        wave[t]=[s.g for s in m.synlist_i]
        bg=np.asarray(batch.columns(bgname,bgcols));ja+=np.bincount(bgidx,weights=-bg[:,0]*bg[:,1],minlength=n)
        js=ja.copy()
        ga,gn,C,rho,w,ea,en=np.asarray(batch.columns(exname,excols)).T
        vv=old[exidx];q=1/(1+C*np.exp(-rho*vv));q1=1/(1+C*np.exp(-rho*(vv+.001)))
        j0=w*(-ga+gn*rho*q*(1-q)*(en-vv)-gn*q);j1=w*(-ga+gn*rho*q1*(1-q1)*(en-vv-.001)-gn*q1)
        sec=-w*(ga*.001+gn*(q1*(vv+.001-en)-q*(vv-en)))/.001
        ja+=np.bincount(exidx,weights=j0,minlength=n);js+=np.bincount(exidx,weights=sec,minlength=n);djs+=np.bincount(exidx,weights=(j1-j0)/.001,minlength=n)
        outward_slope=np.zeros(n);rhs_v=1+D1*(ja-js+djs*dv);rhs_c=np.zeros(n)
        calcium_v=np.zeros(n);calcium_c=np.zeros(n);calcium_i=np.zeros(n);calcium_gate=np.zeros(K)
        for name,spec in specs.items():
            idx=spec['idx'];tp=spec['type'];ion=spec['ion'];ns=spec['ns'];ar=np.asarray(batch.columns(name,spec['columns']));ar={k:ar[:,j] for j,k in enumerate(spec['names'])}
            if tp=='ghk':
                ci,co=(cai[idx],cao[idx]) if ion=='ca' else (cali[idx],calo[idx])
                F,Fv,Fc=ghk_parts(old[idx],ci,co,m.meta['temperature']);F1,Fv1,Fc1=ghk_parts(old[idx]+.001,ci,co,m.meta['temperature'])
                secF=(F1-F)/.001;secFv=(Fv1-Fv)/.001;secFc=(Fc1-Fc)/.001
                pref=ar['i'+ion]/F
                outward_slope[idx]+=pref*secF
                rhs_v[idx]-=D2[idx]*pref*(Fv-secF+secFv*dv[idx])
                if ion=='ca':
                    rhs_c[idx]-=D2[idx]*pref*(Fc+secFc*dv[idx]);calcium_v[idx]+=pref*Fv;calcium_c[idx]+=pref*Fc;calcium_i[idx]+=ar['ica']
                for g,sl in zip(['m','h'][:ns],spec['slots']):
                    pg=ar['did'+g]/F;gcoef[t,0,sl]=-D2[idx]*pg*(F+secF*dv[idx])
                    if ion=='ca':calcium_gate[sl]=pg*F
            else:
                conductance=ar['gna' if ion=='na' else 'gk'];outward_slope[idx]+=conductance
                gates=['o'] if tp in ['sk','kca'] else ['m','h'][:ns]
                for g,sl in zip(gates,spec['slots']):gcoef[t,0,sl]=D2[idx]*(ar['e'+ion]-nxt[idx])*ar['dgd'+g]
            gates=['o'] if tp in ['sk','kca'] else ['m','h'][:ns]
            for g,sl in zip(gates,spec['slots']):
                gcoef[t,2,sl]=ar['d'+g+'d'+g+'_']
                if tp!='sk':gcoef[t,3,sl]=ar['d'+g+'dv']
                if tp in ['sk','kca']:gcoef[t,4,sl]=ar['d'+g+'dc']
        beta=np.zeros(n);alpha=np.zeros(n)
        beta[cadidx]=-(1-decay)*taur*drive/(2*96485.33212*.1)*(calcium_i[cadidx]<0)
        alpha[cadidx]=decay-(1-decay)*taur*pump*1e-4*1e-4/(cai[cadidx]+1e-4)**2
        vcoef[t,0]=rhs_v;vcoef[t,1]=rhs_c;vcoef[t,2]=alpha+beta*calcium_c;vcoef[t,3]=beta*calcium_v
        gcoef[t,1]=calcium_gate*beta[segments]
        d=np.diag(matrix).copy();d[:n]+=D2*outward_slope-D1*js;diag[t]=d
    factor=np.asarray(batch.factor_tree(diag,order,parent,lo,up))
    assert np.all(np.isfinite(diag)) and np.min(abs(diag))>1e-12
    return {'n':n,'nn':nn,'K':K,'T':T,'soma':soma,'parent':parent,'order':order,'lower':lo,'upper':up,'vcoef':vcoef,'gcoef':gcoef,'diag':diag,'factor':factor,'segments':segments,'vnext':vnext,'wave':wave,'D1':D1,'trace':trace,'collect_seconds':time.perf_counter()-tic}
