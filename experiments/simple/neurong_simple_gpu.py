"""SIMPLE forward replicas using NeuronG's native GPU mechanisms."""
from pathlib import Path
import argparse, json, time, os
import numpy as np
import neurong
from biophys_prepare import BiophysPrepare

ROOT=Path(__file__).resolve().parent

def build(setting, case_id, batch):
    started=time.perf_counter()
    locations=np.load(ROOT/'results_exc6/inputs/exc_locations.npy')[case_id]
    times=np.load(ROOT/'results_exc6/inputs/exc_times_ms.npy')[case_id]
    soma=neurong.section('soma','soma');soma.nseg=1;soma.L_um=20.;soma.diam_um=20.
    dend=neurong.section('dend','dend');dend.nseg=3;dend.L_um=60.;dend.diam_um=1.;dend.connect(soma,.5)
    template={'name':'SIMPLE','num_cells':batch,'sections':[soma,dend]}
    logical_segments=list(np.repeat(np.arange(4),5))+locations.tolist()
    instances={'soma':[5.], 'dend':[float(5+np.sum(locations==i)) for i in range(1,4)]}
    weights={'soma':[0.]*5,'dend':sum(([0.]*5+[.001]*int(np.sum(locations==i)) for i in range(1,4)),[])}
    p=BiophysPrepare();allsec=p.for_label('all')
    allsec.resistivity(100.);allsec.specificCapacitance(1.)
    allsec.insert('pas',ion='non_specific',g_pas=.00003,e_pas=-65.)
    active=p.for_label('soma' if setting=='soma_hh' else 'all')
    active.insert('hh',ion='na',erev=50.)
    active.insert('hh',ion='k',erev=-77.,gnabar=.12,gkbar=.036,gl=0.,el=-65.)
    allsec.insert('syn_record',ion='non_specific',instances=instances,w=weights,r_na=1.,
                  AMPA_tau1=.1,AMPA_tau2=2.,NMDA_tau1=2.,NMDA_tau2=75.,
                  GABA_tau1=1.,GABA_tau2=5.,AMPA_e=0.,NMDA_e=0.,GABA_e=-70.,NMDA_C=.267,NMDA_rho=.062)
    bio={'template_name':'SIMPLE','inserts':p.prepare_inserts(),'ion_species':p.prepare_ion_species(),
         'specificCapacitance':p.prepare_specificCapacitance(),'resistivity':p.prepare_resistivity(),'v_init':-65.}
    ps=BiophysPrepare();ps.for_label('soma').insert('VecStim',ion='non_specific',instances=11.)
    sources={**bio,'gid_selector':[0],'inserts':bio['inserts']+ps.prepare_inserts(),
             'ion_species':bio['ion_species']+ps.prepare_ion_species()}
    del sources['template_name']
    del bio['template_name']
    bio['gid_selector']=list(range(1,batch))
    sim=neurong.Sim();sim.set_spike_output_enabled(False);sim.set_device('gpu');sim.set_dt(.025)
    sim.set_output_dir(str(ROOT/'gpu_benchmark'))
    print('MORPH',sim.load_celltemplate_morphology([template]),flush=True)
    sim.apply_celltemplate_biophysics(([bio] if batch>1 else [])+[sources],6.3,record_gids=[])
    stim=[sim.resolve_mech_index_by_loc('VecStim',0,'soma',.5,i) for i in range(11)]
    pre=[];gidlist=[];section=[];loc=[];handles=[];vhandles=[]
    for gid in range(batch):
        slots=[0]*4
        for j,segment in enumerate(logical_segments):
            sec='soma' if segment==0 else 'dend';x=.5 if segment==0 else (segment-.5)/3
            pre.append(stim[j%5 if j<20 else 5+j-20]);gidlist.append(gid)
            section.append(0 if segment==0 else 1);loc.append(x)
            index=sim.resolve_mech_index_by_loc('syn_record',gid,sec,x,slots[segment]);slots[segment]+=1
            if j<20:handles.append(sim.get_variable_handle('syn_record','w',index))
        node=sim.resolve_node_index_by_loc(gid,'soma',.5)
        vhandles.append(sim.get_variable_handle('global','v',node))
    assert min(handles+vhandles+stim)>=0
    assert sim.load_network_artificial_packed(pre,gidlist,section,loc,[1.]*len(pre),[0.]*len(pre),'syn_record')==0
    assert sim.set_vecstim_sequences(stim,list(range(12)),np.r_[np.linspace(20,60,5),times].tolist())==0
    return sim,handles,vhandles,time.perf_counter()-started

def simulate(sim,handles,vhandles,counts):
    begin=time.perf_counter()
    assert sim.set_variables_by_handles(handles,(-.001*np.asarray(counts)).ravel().tolist())==0
    update=time.perf_counter()-begin
    result=np.empty((4001,len(vhandles)),dtype=np.float32)
    run_start=time.perf_counter()
    assert sim.simulate_output_vs_into(result,vhandles,100.,-65.)==0
    run_seconds=time.perf_counter()-run_start
    assert np.all(np.isfinite(result))
    return result,{'weight_update_s':update,'gpu_simulation_and_download_s':run_seconds,'total_s':time.perf_counter()-begin}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--setting',required=True);p.add_argument('--case',type=int,default=0)
    args=p.parse_args();out=ROOT/'gpu_benchmark';out.mkdir(exist_ok=True)
    neurong.warmup_gpu()
    case=ROOT/'gpu_benchmark/cpu_reference'/args.setting/f'case_{args.case:03d}/rec'
    reference=np.load(case/'reference.npz')
    counts=reference['counts']
    sim,handles,vhandles,build_s=build(args.setting,args.case,len(counts))
    voltage,timing=simulate(sim,handles,vhandles,counts)
    reversed_voltage,_=simulate(sim,handles,vhandles,counts[::-1])
    report={'setting':args.setting,'case':args.case,'host':os.uname().nodename,'pid':os.getpid(),
            'device':'gpu','build_s':build_s,'timing':timing,
            'baseline_max_abs_error_mV':float(np.max(np.abs(voltage[:,0]-reference['voltage'][:,0]))),
            'solution_max_abs_error_mV':float(np.max(np.abs(voltage[:,1]-reference['voltage'][:,1]))),
            'all_max_abs_error_mV':float(np.max(np.abs(voltage-reference['voltage']))),
            'classification_mismatches':int(np.sum((voltage.max(axis=0)>=0)!=(reference['voltage'].max(axis=0)>=0))),
            'count':len(counts),
            'updated_batch_max_abs_error_mV':float(np.max(np.abs(reversed_voltage-reference['voltage'][:,::-1]))),
            'gpu_peaks_mV':voltage.max(axis=0).tolist(),
            'reference_peaks_mV':reference['voltage'].max(axis=0).tolist()}
    np.savez_compressed(out/f'parity_{args.setting}_{args.case}.npz',gpu=voltage,cpu=reference['voltage'],counts=counts)
    (out/f'parity_{args.setting}_{args.case}.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
