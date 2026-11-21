"""Pinned L5 model and saved inputs, no dependency on other model folders."""
from pathlib import Path
import numpy as np
from neuron import h
import torch
from GradNeuron_torch_mod import GradNeuron
from L5PC import L5PCcell
HERE=Path(__file__).resolve().parent
MECHS=['Ca_LVAst','Ca_HVA','Nap_Et2','NaTa_t','NaTs2_t','K_Pst','K_Tst','Ih','Im','SKv3_1','hh_grad','SK_E2','CaDynamics_E2']
def build(regime='probe',case=0,temperature=None,candidates=None,weights=None,backend='v2',variant='pulse'):
    temperature_override=temperature is not None
    h.load_file('stdrun.hoc');h.cvode_active(0);h.secondorder=0;h.dt=.025;h.steps_per_ms=40;h.usetable_hh_grad=0
    if hasattr(h,'usetable_hh'):h.usetable_hh=0
    torch.set_num_threads(1)
    temperature_before=float(h.celsius)
    cell=L5PCcell(True,3,choice='area',type='uni')
    temperature_loaded=float(h.celsius)
    if temperature is not None:
        assert regime=='probe','Temperature override prohibited for saved experiments'
        h.celsius=temperature
    temperature=float(h.celsius)
    seg=[s for sec in cell.all for s in sec];n=len(seg)
    distance=np.array([h.distance(cell.soma[0](.5),s) for s in seg])
    if regime=='probe':
        loc=np.array([0,400,600]);events=[np.array([5.]),np.array([5.5]),np.array([6.])];counts=[0,3,0];bw=np.full(3,.2);ratio=[1.,.5];stop=40.;grid=np.array([6.,7.])
        if candidates is None:candidates=[(0,6.),(n-2,7.)]
        old_v=None;old_ans=None
    else:
        d=np.load(HERE/'input'/f'{regime}_{case}.npz');loc=d['loc'];off=d['event_offsets'];events=[d['event_times'][a:b] for a,b in zip(off[:-1],off[1:])]
        counts={'ampa':[2200,2200,550],'nmda':[600,600,150],'bhigh':[600,800,150],'sync':[600,200,150]}[regime]
        ratio=[2.,2.] if regime=='ampa' else [1.,.5]
        bw=np.full(len(loc),.001)
        if regime=='sync':bw[700:800]=.004
        grid=np.linspace(100,400,31);old_v=d['old_v'];old_ans=d['old_ans'];stop=(len(old_v)-1)*.025 if len(old_v) else 400.025
        if candidates is None:
            candidates=[(0,float(t)) for t in grid]
            if variant=='train':
                distal=int(np.flatnonzero(distance>=50)[0])
                candidates += [(distal,float(t)) for t in grid]
    cidx=np.array([int(c[0]) for c in candidates]);ct=np.array([float(c[1]) for c in candidates]);N=len(cidx)
    syn={'ratio':ratio,'AMPA_tau1':.1,'AMPA_tau2':2.,'NMDA_tau1':2.,'NMDA_tau2':75.,'E_exc':0.,'GABAa_tau1':1.,'GABAa_tau2':5.,'GABAb_tau1':5.,'GABAb_tau2':50.,'E_inh':-70.}
    cfg={'record':False,'spiny':False,'N_e':0,'N_i':N,'N_back':counts,'grid_num':max(1,N),'syn_data':syn,'inh_setting':{'if_kernel':False,'if_axon':False,'if_fs':False,'if_slow':False},'dtype':'double','device':'cpu','L_min':1e-6,'L_max':10.,'D_min':1e-6,'D_max':10.,'DH_min':1e-6,'DH_max':10.,'w_i_min':0.,'w_i_max':1e3,'G_max':1e3,'exc_loc_idx':np.array([],int),'inh_loc_idx':cidx,'back_loc_idx':loc,'w_e':np.array([]),'w_i':np.zeros(N) if weights is None else np.array(weights),'w_back':bw,'cell_info':{'cell':cell,'soma':cell.soma[0](.5),'active':True,'v_r':-80.,'mech_list':MECHS}}
    if variant=='train':cfg['inh_setting'].update(if_fs=True,if_depress=True,distance=50)
    out=HERE/'results';out.mkdir(exist_ok=True)
    m=GradNeuron(cfg,str(out)+'/');m.nerst_const_ca=1000*8.31446261815324*(273.15+temperature)/(2*96485.33212)
    m.meta={'regime':regime,'case':case,'temperature':temperature,'temperature_before_load':temperature_before,'temperature_after_load':temperature_loaded,'temperature_override':temperature_override,'v_r':-80.,'stop':stop,'grid':grid,'old_v':old_v,'old_ans':old_ans,'candidates':candidates}
    m.meta['variant']=variant;m.distance=distance;m.probe_count=N;m.active_syn={}
    m.events=events;m.inh_events=[inhibitory_times(variant,distance[int(i)],float(t)) for i,t in candidates]
    return m

def inhibitory_times(variant,distance,t):
    if variant=='pulse':return np.array([t])
    assert variant=='train'
    return np.arange(t,401.,20. if distance<50 else 200.)

def add_event(m,candidate_id,increment=.001):
    candidate_id=int(candidate_id);loc,ti=divmod(candidate_id,31)
    assert 0<=loc<m.Nseg_no_axon and 0<=ti<31
    if candidate_id in m.active_syn:
        i=m.active_syn[candidate_id];m.w_i[i]+=increment;m.synlist_i[i].w=float(m.w_i[i]);return
    segment=m.seglist[loc];variant=m.meta['variant'];near=m.distance[loc]<50
    if variant=='train' and near:
        s=h.tmgExSyn(segment);s.tau_r=.2;s.tau_d=8.3;s.Fac=21;s.Use=.25;s.Dep=706
    else:
        s=h.Exp2Syn_Inhibitory(segment);s.tau1=1.;s.tau2=10. if variant=='train' else 5.
    s.e=-70.;s.w=increment;s.id=m.N_i
    stim=h.VecStim();nc=h.NetCon(stim,s);nc.delay=0;nc.threshold=0;nc.weight[0]=1.
    m.active_syn[candidate_id]=m.N_i
    m.synlist_i.append(s);m.stimlist_i.append(stim);m.nclist_i.append(nc)
    m.inh_events.append(inhibitory_times(variant,m.distance[loc],float(m.meta['grid'][ti])))
    m.w_i=np.append(m.w_i,increment);m.N_i+=1

def candidate_arrays(m):
    loc=np.repeat(np.arange(m.Nseg_no_axon,dtype=np.int32),31)
    oi=np.tile(np.arange(31,dtype=np.int32),m.Nseg_no_axon)
    if m.meta['variant']=='train':oi+=31*(m.distance[loc]>=50)
    return loc,oi
def initialize(m,weights=None):
    if weights is not None:
        m.w_i[:]=weights
        for syn,w in zip(m.synlist_i,weights):syn.w=float(w)
    m.set_stim([],m.inh_events,m.events,[[0,0,0]])
    h.dt=.025;h.finitialize(m.v_r);h.fcurrent()
def forward(m,weights=None):
    initialize(m,weights);v=[m.soma.v]
    for _ in range(round(m.meta['stop']/h.dt)):
        h.fadvance();v.append(m.soma.v)
    return np.array(v)
