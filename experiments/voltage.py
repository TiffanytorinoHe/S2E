"""Full-state autonomous voltage iteration and independent NEURON reference."""
from pathlib import Path
import argparse,sys,os,time,copy,traceback,re,json
import numpy as np
from common import ROOT,group,identity as make_identity,setup,sha,save as save_json
p=argparse.ArgumentParser();p.add_argument('--group',required=True);p.add_argument('--case',type=int,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
key,cid=args.group,args.case;G=group(key);S=ROOT/'models'/G['runtime'];D=args.output.resolve();D.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT/'src/voltage'))
identity=make_identity(G,cid)
def save(name,x):save_json(D/name,x)
def run():
    if (D/'complete.json').exists():
        assert json.loads((D/'identity.json').read_text())==identity
        return
    assert not (D/'error.json').exists(),'Preserved failure; inspect before a fresh run'
    m,model=setup(G,cid,D)
    from neuron import h
    model.initialize(m)
    from engine import Engine
    from synapse_events import independent_waves
    import data_c2py_v2 as batch
    eng=Engine(m,S,G['model']);n=eng.n;dt=eng.dt;T=round(m.meta['stop']/dt);soma=m.seg2idx[m.soma]
    state0=eng.snapshot();waves=independent_waves(m,T,dt,n)
    # Every native table flag is recorded; only HH tables are disabled by model.py.
    settings=dict(dt_ms=dt,celsius=eng.temp,initial_voltage_mV=m.v_r,secondorder=int(h.secondorder),n_segments=n,steps=T,duration_ms=T*dt,model=G['model'],regime=G['regime'],case=cid,new_msn_config=G['new_msn_config'],counts=m.N_back,hh_tables={k:float(getattr(h,k)) for k in ['usetable_hh','usetable_hh_grad'] if hasattr(h,k)},mechanism_tables={name:s['p']['usetable'] for name,s in eng.channels.items()},thresholds=dict(max_abs_voltage_mV=1e-3,rmse_all_segments_mV=1e-4,spike_grid_match=True),definition='V, all active gates, Ca pools and synaptic states evolve solely from their own preceding values; NEURON used only for initial state/parameters and reference error measurement.')
    save('identity.json',identity);save('settings.json',settings)
    def cols(stem,names):
        txt=(S/'x86_64'/(stem+'.cpp')).read_text();ff={k:int(v) for k,v in re.findall(r'^#define (\w+)_columnindex (\d+)$',txt,re.M)}
        return np.array([ff[k] for k in names],np.int32)
    exc='Exp2Syn_Excitatory_back';inh='Exp2Syn_back'
    ec=cols('exp2syn_exc_back',['AMPA_g','NMDA_g','w','NMDA_C','NMDA_rho']);ic=cols('exp2syn_back',['g','w'])
    ei=m.hines2idx[batch.get_ml_nodeindices(exc)];ii=m.hines2idx[batch.get_ml_nodeindices(inh)]
    ed=batch.columns(exc,ec);assert np.all(ed[:,3]==.267) and np.all(ed[:,4]==.062)
    ref=np.zeros((T+1,n));ref[0]=state0[0];samples=[];max_wave=0.;start=time.perf_counter();cal_ref0=state0[2]
    for k in range(T):
        audit=k<4 or k%200==0
        if audit:st=eng.snapshot()
        h.fadvance();ref[k+1]=[s.v for s in m.seglist]
        if audit:
            new=eng.snapshot();samples.append((k,st,new))
            ed=batch.columns(exc,ec);bd=batch.columns(inh,ic)
            rw=np.array([np.bincount(ei,weights=ed[:,0]*ed[:,2],minlength=n),np.bincount(ei,weights=ed[:,1]*ed[:,2],minlength=n),np.bincount(ii,weights=bd[:,0]*bd[:,1],minlength=n)])
            max_wave=max(max_wave,float(abs(rw-waves[k]).max()))
            if G['model']=='msn':assert np.array_equal(new[2],cal_ref0),'Frozen cali unexpectedly changed'
        if k%4000==0:print(key,cid,'reference',k,T,flush=True)
    reference_seconds=time.perf_counter()-start
    teacher=[]
    for k,st,new in samples:
        calc=eng.step(*copy.deepcopy(st),waves[k]);gc={name:float(abs(calc[3][name]-new[3][name]).max()) for name in calc[3]}
        teacher.append(dict(step=k+1,V_mV=float(abs(calc[0]-new[0]).max()),ca_mM=float(abs(calc[1]-new[1]).max()),gates=gc))
    save('one_step.json',teacher)
    print('ONE_STEP',key,cid,max(q['V_mV'] for q in teacher),max(q['ca_mM'] for q in teacher),{name:max(q['gates'][name] for q in teacher) for name in eng.channels},'WAVE',max_wave,flush=True)
    # Destroy reference handles before the autonomous integration.
    eng.m=None;eng.batch=None
    sites=np.unique(np.r_[soma,np.linspace(0,n-1,12,dtype=int),[np.argmin(abs(m.distance-d)) for d in np.quantile(m.distance,[0,.25,.5,.75,1])]]).astype(int)
    save('geometry.json',dict(soma_index=int(soma),site_indices=sites.tolist(),distance_um=m.distance.tolist(),section_names=[s.sec.name() for s in m.seglist],segment_x=[s.x for s in m.seglist]))
    st=copy.deepcopy(state0);soma_v=np.zeros(T+1);soma_v[0]=st[0][soma];trace=np.zeros((T+1,len(sites)));trace[0]=st[0][sites]
    max_t=np.zeros(T+1);rms_t=np.zeros(T+1);max_seg=np.zeros(n);ss_seg=np.zeros(n);argmax=dict(step=0,segment=0,error_mV=0.);ca_max=0.;gate_max=0.;sample_map={k+1:new for k,_,new in samples}
    start=time.perf_counter()
    for k in range(T):
        st=eng.step(*st,waves[k]);v=st[0]
        assert np.all(np.isfinite(v)) and np.all(st[1]>0),('nonfinite state',k)
        err=v-ref[k+1];ae=abs(err);max_t[k+1]=ae.max();rms_t[k+1]=np.sqrt(np.mean(err**2));max_seg=np.maximum(max_seg,ae);ss_seg+=err**2
        if max_t[k+1]>argmax['error_mV']:argmax=dict(step=k+1,segment=int(np.argmax(ae)),error_mV=float(ae.max()))
        soma_v[k+1]=v[soma];trace[k+1]=v[sites]
        if k+1 in sample_map:
            nxt=sample_map[k+1];ca_max=max(ca_max,float(abs(st[1]-nxt[1]).max()));gate_max=max(gate_max,max(float(abs(st[3][name]-nxt[3][name]).max()) for name in st[3]))
        if k%4000==0:print(key,cid,'autonomous',k,T,flush=True)
    elapsed=time.perf_counter()-start;t=np.arange(T+1)*dt
    def sp(v):return t[1:][(v[:-1]<0)&(v[1:]>=0)]
    a=sp(soma_v);b=sp(ref[:,soma]);same=np.array_equal(a,b);serr=soma_v-ref[:,soma]
    result=dict(identity=identity,all_segments_max_abs_mV=float(max_t.max()),all_segments_rmse_mV=float(np.sqrt(ss_seg.sum()/((T+1)*n))),soma_max_abs_mV=float(abs(serr).max()),soma_rmse_mV=float(np.sqrt(np.mean(serr**2))),reference_spikes_ms=b.tolist(),rollout_spikes_ms=a.tolist(),spike_grid_identical=bool(same),spike_time_max_abs_ms=float(abs(a-b).max()) if len(a)==len(b) and len(a) else (0. if len(a)==len(b) else None),sampled_ca_max_abs_mM=ca_max,sampled_gate_max_abs=gate_max,synapse_sampled_max_abs_uS=max_wave,one_step_max_mV=max(q['V_mV'] for q in teacher),worst_error=argmax,reference_seconds=reference_seconds,autonomous_seconds=elapsed)
    result['passed']=bool(result['all_segments_max_abs_mV']<=1e-3 and result['all_segments_rmse_mV']<=1e-4 and same and max_wave<1e-10)
    result['windows']=[dict(start_ms=float(u),end_ms=float(min(u+50,t[-1])),max_all_segments_mV=float(max_t[(t>=u)&(t<=u+50)].max())) for u in np.arange(0,t[-1],50)]
    save('result.json',result)
    np.savez_compressed(D/'traces.npz',time_ms=t,neuron_soma=ref[:,soma],autonomous_soma=soma_v,site_indices=sites,neuron_sites=ref[:,sites],autonomous_sites=trace,max_error_over_segments_mV=max_t,rmse_over_segments_mV=rms_t,max_error_per_segment_mV=max_seg,rmse_per_segment_mV=np.sqrt(ss_seg/(T+1)))
    save('complete.json',dict(passed=result['passed'],result_sha256=sha(D/'result.json'),traces_sha256=sha(D/'traces.npz'),finished_unix=time.time()))
    print('RESULT',key,cid,result['passed'],result['all_segments_max_abs_mV'],result['all_segments_rmse_mV'],b.tolist(),a.tolist(),flush=True)
if __name__=='__main__':
    try:run()
    except BaseException:
        save('error.json',dict(traceback=traceback.format_exc(),unix=time.time()));raise
