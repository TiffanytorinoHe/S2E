"""Historical FIG2_D inputs only; native model loaders and original parameters."""
from pathlib import Path
import json, os
import numpy as np
import torch
from neuron import h
from GradNeuron_torch_mod import GradNeuron
HERE=Path(__file__).resolve().parent
MODEL=json.loads((HERE/'model_config.json').read_text())['model']
MECHS_L23=['Ca_LVAst','Ca_HVA','Nap_Et2','NaTa_t','NaTs2_t','NaTg','K_Pst','K_Tst','Ih','Im','SKv3_1','hh_grad','SK_E2','CaDynamics_E2']
MECHS_MSN=['naf_kai','kaf_kai','kas_kai','kdr_kai','kir_kai','bk','sk_kai','car_kai','can_kai','caq_kai','cal12','cal13','cat32','cat33','cadyn','caldyn']

def build(regime='ampa',case=0,variant='pulse'):
    assert variant=='pulse','Only the historical FIG2_D single-event experiment is authorized here'
    h.load_file('stdrun.hoc');h.load_file('import3d.hoc');h.cvode_active(0);h.secondorder=0;h.dt=.025;h.steps_per_ms=40
    for name in ['usetable_hh','usetable_hh_grad']:
        if hasattr(h,name):setattr(h,name,0)
    torch.set_num_threads(1)
    before=float(h.celsius)
    if MODEL=='L23':
        from L23 import L23PC
        cell=L23PC(neuron_num=9,max_segment_len=10,choice='area',type='uni')
        counts={'ampa':[800,800,200],'nmda':[240,240,60],'bhigh':[200,200,50],'sync':[200,200,50]}[regime]
        mechs=MECHS_L23;keep=[]
    else:
        assert MODEL=='MSN'
        from MSN import MSN
        cell=MSN(params='./params_dMSN.json',morphology='latest_WT-P270-20-14ak.swc')
        casc=h.D1_reduced_cascade2_0(.5,sec=cell.soma[0]);pointer=casc._ref_Target1p
        for sec in h.allsec():
            for s in sec:
                for name in ['kas_kai','naf_kai']:h.setpointer(pointer,'pka',getattr(s,name))
                if 'axon' not in sec.name():
                    for name in ['kaf_kai','cal12','cal13','kir_kai']:h.setpointer(pointer,'pka',getattr(s,name))
                    if 'soma' in sec.name():h.setpointer(pointer,'pka',s.can_kai)
        keep=[casc]
        counts={'ampa':[2400,2400,600],'nmda':[600,600,150],'bhigh':[400,400,100],'sync':[200,200,50]}[regime]
        mechs=MECHS_MSN
    temperature=float(h.celsius)
    seg=[s for sec in cell.all for s in sec];distance=np.array([h.distance(cell.soma[0](.5),s) for s in seg])
    d=np.load(HERE/os.environ['CAL_INPUT']);counts=d['counts'].tolist();loc=d['loc'];off=d['event_offsets'];events=[d['event_times'][a:b] for a,b in zip(off[:-1],off[1:])]
    assert len(loc)==sum(counts)==len(events)
    grid=np.linspace(100,400,31);cidx=np.zeros(31,dtype=int);bw=np.full(len(loc),.001)
    # Historical workspace FIG2_D uses .001 uS. The newly supplied L23 entry's
    # .0004 setting fails its saved baseline; .001 reproduces it to 8e-11 mV.
    if regime=='sync':bw[sum(counts[:2])-int(d['n_sync']):sum(counts[:2])]=.004
    syn={'ratio':[2.,2.] if regime=='ampa' else [1.,.5],'AMPA_tau1':.1,'AMPA_tau2':2.,'NMDA_tau1':2.,'NMDA_tau2':75.,'E_exc':0.,'GABAa_tau1':1.,'GABAa_tau2':5.,'GABAb_tau1':5.,'GABAb_tau2':50.,'E_inh':-70.}
    cfg={'record':False,'spiny':False,'N_e':0,'N_i':31,'N_back':counts,'grid_num':31,'syn_data':syn,'inh_setting':{'if_kernel':False,'if_axon':False,'if_fs':False,'if_slow':False},'dtype':'double','device':'cpu','L_min':1e-6,'L_max':10.,'D_min':1e-6,'D_max':10.,'DH_min':1e-6,'DH_max':10.,'w_i_min':0.,'w_i_max':1e3,'G_max':1e3,'exc_loc_idx':np.array([],int),'inh_loc_idx':cidx,'back_loc_idx':loc,'w_e':np.array([]),'w_i':np.zeros(31),'w_back':bw,'cell_info':{'cell':cell,'soma':cell.soma[0](.5),'active':True,'v_r':-80.,'mech_list':mechs}}
    out=HERE/'runtime';out.mkdir(exist_ok=True)
    m=GradNeuron(cfg,str(out)+'/');m.keep=keep;m.distance=distance;m.probe_count=31;m.active_syn={}
    m.meta={'model':MODEL,'regime':regime,'case':case,'variant':variant,'temperature':temperature,'temperature_before_load':before,'temperature_after_load':temperature,'v_r':-80.,'stop':float(d['tstop']),'grid':grid,'old_v':d['old_v'],'old_ans':d['old_ans']}
    m.events=events;m.inh_events=[np.array([t]) for t in grid]
    return m

def add_event(m,candidate_id,increment=.001):
    candidate_id=int(candidate_id);loc,ti=divmod(candidate_id,31)
    assert 0<=loc<m.Nseg_no_axon and 0<=ti<31
    if candidate_id in m.active_syn:
        i=m.active_syn[candidate_id];m.w_i[i]+=increment;m.synlist_i[i].w=float(m.w_i[i]);return
    s=h.Exp2Syn_Inhibitory(m.seglist[loc]);s.tau1=1.;s.tau2=5.;s.e=-70.;s.w=increment;s.id=m.N_i
    stim=h.VecStim();nc=h.NetCon(stim,s);nc.delay=0;nc.threshold=0;nc.weight[0]=1.
    m.active_syn[candidate_id]=m.N_i;m.synlist_i.append(s);m.stimlist_i.append(stim);m.nclist_i.append(nc)
    m.inh_events.append(np.array([float(m.meta['grid'][ti])]))
    m.w_i=np.append(m.w_i,increment);m.N_i+=1

def candidate_arrays(m):
    return np.repeat(np.arange(m.Nseg_no_axon,dtype=np.int32),31),np.tile(np.arange(31,dtype=np.int32),m.Nseg_no_axon)

def initialize(m,weights=None):
    if weights is not None:
        m.w_i[:]=weights
        for s,w in zip(m.synlist_i,weights):s.w=float(w)
    m.set_stim([],m.inh_events,m.events,[[0,0,0]])
    h.dt=.025;h.finitialize(m.v_r);h.fcurrent()

def forward(m,weights=None):
    initialize(m,weights);v=[m.soma.v]
    for _ in range(round(m.meta['stop']/h.dt)):
        h.fadvance();v.append(m.soma.v)
    return np.array(v)
