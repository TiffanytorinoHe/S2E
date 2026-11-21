"""State-conditioned, double precision, tree-solve tangent for L5 mechanisms.

Only fixed geometry, fixed event times, weight-scaled ohmic inhibitory sources
are supported. Unsupported channel classes fail explicitly.
"""
import numpy as np
import time,ctypes,re
from pathlib import Path
from neuron import h
import data_c2py as bridge
from myfunc_torch import CalcMatrix
from ChannelType import mech_type
from model import initialize

def tree_structure(matrix,root):
    adjacency=[np.flatnonzero((matrix[i]!=0)&(np.arange(len(matrix))!=i)).tolist() for i in range(len(matrix))]
    assert sum(map(len,adjacency))==2*(len(matrix)-1),'Expected connected tree'
    parent=np.full(len(matrix),-2,np.int32);parent[root]=-1;order=[root]
    for i in order:
        for j in adjacency[i]:
            if j==parent[i]:continue
            assert parent[j]==-2,'Tree cycle';parent[j]=i;order.append(j)
    assert len(order)==len(matrix)
    order=np.array(order,np.int32)
    lower=np.array([0. if parent[i]<0 else matrix[i,parent[i]] for i in range(len(matrix))])
    upper=np.array([0. if parent[i]<0 else matrix[parent[i],i] for i in range(len(matrix))])
    return parent,order,lower,upper

def collect(m):
    tic=time.perf_counter();initialize(m)
    matrix=CalcMatrix(m.seclist,[{},{},{},{}]).numpy();n=m.Nseg;nn=len(matrix);T=round(m.meta['stop']/h.dt)
    soma=m.seg2idx[m.soma]
    parent,order,lo,up=tree_structure(matrix,soma)
    D1=np.array([h.dt*1e5/s.cm/s.area() for s in m.seglist]);D2=np.array([h.dt*1e3/s.cm for s in m.seglist])
    index={name:m.rec[name]['idx'].cpu().numpy().astype(int) for name in m.mech_list}
    slots=[];segments=[]
    for name in m.mech_list:
        tp,ion,ns=mech_type[name]
        assert tp in ['cad','hh_ori','hh_like','sk'],f'Unsupported mechanism {name}'
        if tp=='cad':continue
        for k in range(ns):
            for idx in index[name]:slots.append((name,k,int(idx)));segments.append(int(idx))
    K=len(slots);segments=np.array(segments,np.int32)
    # v=[voltage_old, calcium_old, calcium_old->new, voltage_old->Ca_new]
    vcoef=np.zeros((T,4,n));gcoef=np.zeros((T,5,K));diag=np.zeros((T,nn))
    vnext=np.zeros((T,n));wave=np.zeros((T,m.N_i));trace=np.zeros(T+1);trace[0]=m.soma.v
    # Nernst dependence only where NEURON recomputes reversal from concentrations.
    cadidx=index.get('CaDynamics_E2',np.array([],int));theta=np.zeros(n)
    for i in cadidx:
        s=m.seglist[i];style=int(h.ion_style('ca_ion',sec=s.sec))
        if (style//64)%2:theta[i]=s.eca/np.log(s.cao/s.cai)
    slotmap={}
    for si,(name,k,idx) in enumerate(slots):slotmap.setdefault(name,{}).setdefault(k,[]).append(si)
    import data_c2py_v2 as batch
    cpp=(Path(__file__).resolve().parent/'x86_64/exp2syn_exc_back.cpp').read_text()
    fields={k:int(v) for k,v in re.findall(r'^#define (\w+)_columnindex (\d+)$',cpp,re.M)}
    cols=np.array([fields[k] for k in ['AMPA_g','NMDA_g','NMDA_C','NMDA_rho','w','AMPA_e','NMDA_e']],np.int32)
    excidx=m.hines2idx[batch.get_ml_nodeindices('Exp2Syn_Excitatory_back')]
    assert np.all(excidx>=0)
    for t in range(T):
        old=np.array([s.v for s in m.seglist]);oldca=np.ones(n)
        for i in cadidx:oldca[i]=m.seglist[i].cai
        h.fadvance();nxt=np.array([s.v for s in m.seglist]);vnext[t]=nxt;trace[t+1]=m.soma.v;dv=nxt-old
        raw={name:np.asarray(bridge.mech_data_rec(name)).reshape(len(index[name]),-1) for name in m.mech_list if len(index[name])}
        # Exact numerical current slope and its old-voltage derivative.
        ja=np.zeros(n);js=np.zeros(n);djs=np.zeros(n)
        for s in m.synlist_i:
            i=m.seg2idx[s.get_segment()];ja[i]-=s.w*s.g;js[i]-=s.w*s.g
        wave[t]=[s.g for s in m.synlist_i]
        # Vectorized bridge arrays avoid one Python call for every background synapse.
        for name in ['Exp2Syn_Excitatory_back','Exp2Syn_back']:
            data=np.asarray(bridge.syn_data_rec(name)).reshape(-1,4)
            if len(data):ja+=data[m.idx2hines,2]
        # secant correction only needed for voltage-dependent NMDA current.
        js=ja.copy()
        ga,gn,C,rho,w,ea,en=batch.columns('Exp2Syn_Excitatory_back',cols).T
        vv=old[excidx];q=1/(1+C*np.exp(-rho*vv));q1=1/(1+C*np.exp(-rho*(vv+.001)))
        j0=w*(-ga+gn*rho*q*(1-q)*(en-vv)-gn*q)
        j1=w*(-ga+gn*rho*q1*(1-q1)*(en-vv-.001)-gn*q1)
        sec=-w*(ga*.001+gn*(q1*(vv+.001-en)-q*(vv-en)))/.001
        js+=np.bincount(excidx,weights=sec-j0,minlength=n)
        djs+=np.bincount(excidx,weights=(j1-j0)/.001,minlength=n)
        g=np.zeros(n);gca=np.zeros(n);gatev=np.zeros(K);gateca=np.zeros(K)
        for name,ar in raw.items():
            tp,ion,ns=mech_type[name];idx=index[name]
            if tp=='cad':continue
            if tp=='hh_ori':
                g[idx]+=ar[:,2]+ar[:,3];dg=ar[:,4:7];rev=[ar[:,-2],ar[:,-2],ar[:,-1]];beta=ar[:,7:10];alpha=ar[:,10:13]
            else:
                g[idx]+=ar[:,1];dg=ar[:,2:2+ns];rev=[ar[:,-1]]*ns;beta=ar[:,2+ns:2+2*ns];alpha=ar[:,2+2*ns:2+3*ns]
            for k in range(ns):
                sl=np.array(slotmap[name][k]);gatev[sl]=D2[idx]*(rev[k]-nxt[idx])*dg[:,k]
                gcoef[t,2,sl]=alpha[:,k]
                if tp=='sk':
                    ca=np.array([m.seglist[i].cai for i in idx]);q=1/(1+(.00043/ca)**4.8)
                    gcoef[t,4,sl]=(1-np.exp(-h.dt))*4.8*q*(1-q)/ca
                else:gcoef[t,3,sl]=beta[:,k]
                if ion=='ca':gateca[sl]=(old[idx]-rev[k])*dg[:,k]
            if ion=='ca':gca[idx]+=ar[:,1]
        betaC=np.zeros(n);alphaC=np.zeros(n)
        if len(cadidx):
            alphaC[cadidx]=raw['CaDynamics_E2'][:,2];betaC[cadidx]=raw['CaDynamics_E2'][:,1]
        eprime=-theta/oldca
        vcoef[t,0]=1+D1*(ja-js+djs*dv);vcoef[t,1]=D2*gca*eprime
        vcoef[t,2]=alphaC-betaC*gca*eprime;vcoef[t,3]=betaC*gca
        gcoef[t,0]=gatev;gcoef[t,1]=gateca*betaC[segments]
        d=np.diag(matrix).copy();d[:n]+=D2*g-D1*js
        diag[t]=d
    factor=np.asarray(batch.factor_tree(diag,order,parent,lo,up))
    assert np.all(np.isfinite(diag)) and np.min(abs(diag))>1e-12
    result={'n':n,'nn':nn,'K':K,'T':T,'soma':soma,'parent':parent,'order':order,'lower':lo,'upper':up,'vcoef':vcoef,'gcoef':gcoef,'diag':diag,'factor':factor,'segments':segments,'vnext':vnext,'wave':wave,'D1':D1,'trace':trace,'collect_seconds':time.perf_counter()-tic}
    return result

def cpu_reference(c,loc,onset_index,record=True):
    tic=time.perf_counter();P=len(loc);loc=np.array(loc,int);oi=np.array(onset_index,int);n=c['n'];nn=c['nn'];K=c['K'];T=c['T'];s=c['segments']
    v=np.zeros((nn,P));ca=np.zeros((n,P));gate=np.zeros((K,P));minimum=np.zeros(P);tr=np.zeros((T+1,P)) if record else None
    for t in range(T):
        vc=c['vcoef'][t];gc=c['gcoef'][t]
        rhs=np.zeros_like(v);rhs[:n]=vc[0,:,None]*v[:n]+vc[1,:,None]*ca
        ca=vc[2,:,None]*ca+vc[3,:,None]*v[:n]
        np.add.at(rhs,s,gc[0,:,None]*gate);np.add.at(ca,s,gc[1,:,None]*gate)
        rhs[loc,np.arange(P)]+=c['D1'][loc]*c['wave'][t,oi]*(-70-c['vnext'][t,loc])
        for j in c['order'][:0:-1]:rhs[c['parent'][j]]-=c['factor'][t,j]*rhs[j]
        root=c['order'][0];rhs[root]/=c['diag'][t,root]
        for j in c['order'][1:]:rhs[j]=(rhs[j]-c['lower'][j]*rhs[c['parent'][j]])/c['diag'][t,j]
        v=rhs;gate=gc[2,:,None]*gate+gc[3,:,None]*v[s]+gc[4,:,None]*ca[s]
        minimum=np.minimum(minimum,v[c['soma']])
        if record:tr[t+1]=v[c['soma']]
    return {'min':minimum,'trace':tr,'seconds':time.perf_counter()-tic}

def gpu(c,loc,oi,record=False):
    lib=ctypes.CDLL(str(Path(__file__).resolve().parent/'libtangent.so'))
    fn=lib.tangent_gpu;fn.restype=ctypes.c_int
    loc=np.ascontiguousarray(loc,np.int32);oi=np.ascontiguousarray(oi,np.int32);P=len(loc)
    assert P>0 and np.all((loc>=0)&(loc<c['n'])) and np.all((oi>=0)&(oi<c['wave'].shape[1]))
    # Adjacent GPU lanes start at the same time, avoiding work before a pulse.
    permutation=np.argsort(oi,kind='stable');inverse=np.argsort(permutation)
    loc=loc[permutation];oi=oi[permutation]
    output=np.zeros(P);trace=np.zeros((c['T']+1,P)) if record else np.zeros(1);timing=np.zeros(3)
    arrays=[c[k] for k in ['parent','order','lower','segments','vcoef','gcoef','diag','factor','vnext','wave','D1']]+[loc,oi,output,trace,timing]
    arrays=[np.ascontiguousarray(x) for x in arrays]
    args=[ctypes.c_int(x) for x in [c['n'],c['nn'],c['K'],c['T'],P,c['wave'].shape[1],c['soma'],int(record)]]+[ctypes.c_void_p(a.ctypes.data) for a in arrays]
    ret=fn(*args)
    if ret:raise RuntimeError(f'CUDA error {ret}')
    return {'min':arrays[-3][inverse],'trace':arrays[-2][:,inverse] if record else None,'timing_s':arrays[-1].tolist()}
