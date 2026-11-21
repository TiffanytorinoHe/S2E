"""One all-HH case per synchronous GPU batch; atomic prefix checkpoints."""
from pathlib import Path
import argparse, json, os, subprocess, time, hashlib
import numpy as np
from production_common import ROOT,RUN,sha,atomic_json,read_json,inputs_sha
from gt_enumeration import total_below,candidates,next_chunk

def cpu_check(case,counts,folder,label):
    where=folder/'cpu_checks'/label;where.mkdir(parents=True,exist_ok=True)
    inp=where/'counts.npy';np.save(inp,counts)
    with (where/'process.log').open('w') as f:
        subprocess.run(['bash',str(ROOT/'cpu_verify_counts.sh'),str(case),str(inp),str(where/'output')],
            cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    data=np.load(where/'output/all_hh'/f'case_{case:03d}/rec/forward_counts.npz')
    assert int(data['case_id'])==case and not bool(data['hh_table'])
    np.testing.assert_array_equal(data['counts'],counts)
    return data['voltage'].max(axis=0),str(where)

def fingerprint(record):
    spec={'case_id':record['case_id'],'input_sha256':record['input_sha256'],
        'model':'all_hh_direct_hh_dt0.025_t100_vinit-65_exc6_w0.001_ratio1',
        'candidate_grid':'segments0..3 x times20,30,40,50,60; increment0.001',
        'enumeration_order':'increasing-k then itertools.combinations_with_replacement(range(20),k)',
        'model_source_sha256':sha(ROOT/'neurong_simple_gpu.py'),'enumeration_source_sha256':sha(ROOT/'gt_enumeration.py')}
    return hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest(),spec

def run(case,out,batch_cap,max_batches,max_length=10):
    out.mkdir(parents=True,exist_ok=True)
    ready=read_json(RUN/'ready'/f'case_{case:03d}.json')
    assert ready['case_id']==case and ready['setting']=='all_hh' and ready['input_sha256']==inputs_sha()
    assert ready['ans_length']<=max_length, 'Case exceeds explicitly selected length limit'
    mark,spec=fingerprint(ready)
    if (out/'result.json').exists():
        result=read_json(out/'result.json');assert result['fingerprint']==mark;return result
    checkpoint=out/'checkpoint.json'
    if checkpoint.exists():
        state=read_json(checkpoint);assert state['fingerprint']==mark
        position=0
        for entry in state['batches']:
            assert entry['start']==position and sha(out/'chunks'/entry['file'])==entry['sha256']
            position=entry['stop']
        assert position==state['next_ordinal']
    else:
        state={'case_id':case,'fingerprint':mark,'specification':spec,'next_ordinal':0,
            'upper_bound':ready['ans_length'],'upper_ans':ready['ans'],
            'algorithm_source_sha256':ready['algorithm_result_sha256'],
            'reference_baseline_peak_mV':ready['reference_baseline_peak_mV'],
            'reference_final_peak_mV':ready['reference_final_peak_mV'],
            'batches':[],'created_at':time.time(),'job_id':os.environ.get('SLURM_JOB_ID')}
        atomic_json(checkpoint,state)
        atomic_json(out/'source_algorithm.json',ready)
    L=state['upper_bound'];total=total_below(L)
    assert L<=max_length, 'Checkpoint exceeds explicitly selected length limit'
    def finish(length,witness,peak,reason,ordinal=None):
        # Reuse an existing independent CPU certificate for the known witness;
        # new GPU-discovered witnesses receive a fresh CPU forward check.
        counts=np.bincount(witness,minlength=20).reshape(1,20)
        if length==0:
            checked=[state['reference_baseline_peak_mV']];location=str(out/'source_algorithm.json')
        elif witness==state['upper_ans']:
            checked=[state['reference_final_peak_mV']];location=str(out/'source_algorithm.json')
        else:checked,location=cpu_check(case,counts,out,'final_witness')
        assert checked[0]<0, 'GPU witness failed independent CPU verification'
        result={'case_id':case,'setting':'all_hh','status':'complete','minimum_length':length,
            'witness':witness,'cpu_witness_peak_mV':float(checked[0]),'gpu_witness_peak_mV':peak,
            'reason':reason,'first_feasible_ordinal':ordinal,'upper_bound_used':L,
            'evaluated_candidates':state['next_ordinal'],'exhaustive_candidates_below_minimum':total_below(length),
            'fingerprint':mark,'specification':spec,'algorithm_source_sha256_used':state['algorithm_source_sha256'],
            'cpu_verification':location,'completed_at':time.time(),'job_id':os.environ.get('SLURM_JOB_ID'),
            'host':os.uname().nodename,'batch_count':len(state['batches'])}
        assert state['next_ordinal']>=total_below(length)
        atomic_json(out/'result.json',result);print('GT_COMPLETE',json.dumps(result),flush=True)
        return result
    if state['reference_baseline_peak_mV']<0:return finish(0,[],None,'baseline already nonspiking')
    assert L>=1 and state['reference_final_peak_mV']<0
    if L==1:return finish(1,state['upper_ans'],None,'baseline spikes and one-event witness is feasible')
    if state.get('pending_witness'):
        w=state['pending_witness'];return finish(w['length'],w['ans'],w['peak'],'first feasible ordered multiset',w['ordinal'])
    if state['next_ordinal']==total:return finish(L,state['upper_ans'],None,'all strictly shorter multisets spike')
    import neurong
    from neurong_simple_gpu import build,simulate
    neurong.warmup_gpu()
    batch=min(batch_cap,total)
    sim,handles,vhandles,build_s=build('all_hh',case,batch)
    control=np.zeros((batch,20),dtype=np.int16);control[1]=np.bincount(state['upper_ans'],minlength=20)
    v,_=simulate(sim,handles,vhandles,control)
    p=v[:,:2].max(axis=0)
    assert p[0]>=0 and p[1]<0
    np.testing.assert_allclose(p,[state['reference_baseline_peak_mV'],state['reference_final_peak_mV']],atol=.001,rtol=0)
    del v
    state['last_calibration']={'build_s':build_s,'gpu_peaks':p.tolist(),'at':time.time(),'batch_size':batch}
    atomic_json(checkpoint,state)
    stream=candidates(L,state['next_ordinal']);performed=0
    while True:
        started=time.perf_counter();chunk=next_chunk(stream,batch)
        if chunk is None:break
        ranks,ks,counts=chunk;valid=len(ranks);assert ranks[0]==state['next_ordinal']
        v,timing=simulate(sim,handles,vhandles,counts);peaks=v[:,:valid].max(axis=0);del v
        close=np.flatnonzero(np.abs(peaks)<.001)
        if len(close):
            actual,_=cpu_check(case,counts[close],out,f'borderline_{ranks[0]:012d}')
            peaks=peaks.astype(np.float64);peaks[close]=actual
        flags=peaks>=0
        name=f'batch_{ranks[0]:012d}_{ranks[-1]+1:012d}.npz'
        saved=out/'chunks'/name;saved.parent.mkdir(exist_ok=True)
        temp=saved.with_suffix('.tmp')
        with temp.open('wb') as f:np.savez_compressed(f,case_id=case,first_ordinal=ranks[0],last_ordinal=ranks[-1],
            k=ks,peak_mV=peaks,spiked=flags,cpu_corrected_indices=close)
        os.replace(temp,saved)
        state['next_ordinal']=int(ranks[-1]+1)
        state['batches'].append({'file':name,'sha256':sha(saved),'start':int(ranks[0]),'stop':state['next_ordinal'],
            'valid':valid,'padded':batch-valid,'cpu_boundary_checks':len(close),
            'seconds':time.perf_counter()-started,'timing':timing})
        wins=np.flatnonzero(~flags)
        if len(wins):
            j=int(wins[0]);witness=np.repeat(np.arange(20),counts[j]).tolist()
            state['pending_witness']={'length':int(ks[j]),'ans':witness,'peak':float(peaks[j]),'ordinal':int(ranks[j])}
        state['updated_at']=time.time();atomic_json(checkpoint,state)
        print('GT_BATCH',case,state['next_ordinal'],total,'wins',len(wins),flush=True)
        if len(wins):
            w=state['pending_witness'];return finish(w['length'],w['ans'],w['peak'],'first feasible ordered multiset',w['ordinal'])
        performed+=1
        if max_batches and performed>=max_batches:print('CHECKPOINT_TEST_STOP',flush=True);raise SystemExit(75)
    assert state['next_ordinal']==total
    return finish(L,state['upper_ans'],None,'all strictly shorter multisets spike')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',type=int,required=True)
    p.add_argument('--output',type=Path);p.add_argument('--batch',type=int,default=131072)
    p.add_argument('--max-batches',type=int,default=0)
    p.add_argument('--max-length',type=int,default=10);a=p.parse_args()
    out=(a.output or RUN/'groundtruth'/f'case_{a.case:03d}').resolve()
    assert out.is_relative_to(ROOT)
    run(a.case,out,a.batch,a.max_batches,a.max_length)
