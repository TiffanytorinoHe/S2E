"""Real saved-input gates in an isolated process with the frozen model binary."""
import os,sys,json,hashlib,time,traceback,ctypes
from pathlib import Path
import numpy as np
from common import ROOT,group,setup,identity as make_identity
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--group',required=True);parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();key=args.group;dest=args.output.resolve();dest.mkdir(parents=True,exist_ok=True)
R=ROOT/'src/profiles';sys.path.insert(0,str(R))

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,q):
    p=dest/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(q,indent=2));t.replace(p)

def old_forward(c,loc,oi):
    P=len(loc);trace=np.zeros((c['T']+1,P));score=np.zeros(P);timing=np.zeros(3)
    arrays=[c[k] for k in ['parent','order','lower','segments','vcoef','gcoef','diag','factor','vnext','wave','D1']]+[np.array(loc,np.int32),np.array(oi,np.int32),score,trace,timing]
    arrays=[np.ascontiguousarray(x) for x in arrays]
    lib=ctypes.CDLL(str(ROOT/'models/l5pc/libtangent.so'))
    ret=lib.tangent_gpu(*[ctypes.c_int(x) for x in [c['n'],c['nn'],c['K'],c['T'],P,c['wave'].shape[1],c['soma'],1]],*[ctypes.c_void_p(a.ctypes.data) for a in arrays])
    assert ret==0
    return trace,timing.tolist()

def old_reverse(c,loc,target):
    out=np.zeros((c['T'],len(loc)));timing=np.zeros(3)
    arrays=[c[k] for k in ['parent','order','lower','upper','segments','vcoef','gcoef','diag','D1']]+[np.asarray(loc,np.int32),np.asarray(target,np.int32),out,timing]
    arrays=[np.ascontiguousarray(x) for x in arrays]
    lib=ctypes.CDLL(str(ROOT/'models/l5pc/libadjoint.so'))
    ret=lib.adjoint_gpu(*[ctypes.c_int(x) for x in [c['n'],c['nn'],c['K'],c['T'],len(loc),c['soma']]],*[ctypes.c_void_p(a.ctypes.data) for a in arrays])
    assert ret==0
    return out,timing.tolist()

def run():
    from kernel_profiles import Resident,make_profiles
    assert not (dest/'error.json').exists(),'Preserved failure; inspect before a new run'
    g=group(key);identity=make_identity(g,0)
    m,model=setup(g,0,dest)
    from tangent import collect,cpu_reference
    from neuron import h
    baseline=model.forward(m);c=collect(m)
    assert np.array_equal(baseline,c['trace'])
    assert abs(m.meta['stop']-400.025)<1e-8
    report={'identity':identity,'temperature':m.meta['temperature'],'v_r':m.v_r,'baseline_collect_max_error':float(np.max(abs(baseline-c['trace']))),'collect_seconds':c['collect_seconds'],'dimensions':{k:c[k] for k in ['n','nn','K','T']}}
    save('progress.json',dict(report,stage='coefficients_collected'))
    loc=np.unique(np.linspace(0,m.Nseg_no_axon-1,32,dtype=np.int32));oi=np.arange(len(loc),dtype=np.int32)%20
    with Resident(c) as device:
        new=device.forward(loc,oi,True);old,ft=old_forward(c,loc,oi)
        error=float(np.max(abs(old-new['trace'])));assert error<1e-10,error
        target=np.argmin(new['trace'],axis=0).astype(np.int32)
        kernel,at=device.adjoint(loc,target);reference,rt=old_reverse(c,loc,target)
        rel=float(np.max(abs(kernel-reference)/np.maximum(1.,abs(reference))));assert rel<1e-10,rel
        direct=new['trace'][target,np.arange(len(loc))]
        dual=np.sum(kernel*c['wave'][:,oi]*(c['vnext'][:,loc]+70),axis=0)
        dualerr=float(np.max(abs(dual-direct)/np.maximum(1.,abs(direct))));assert dualerr<1e-8
        # Explicit target boundary and duplicate endpoint check.
        boundary=np.array([0,1,64,65,c['T'],c['T']],np.int32);bl=np.array([0,0,0,0,0,int(loc[-1])],np.int32)
        kb,_=device.adjoint(bl,boundary);kr,_=old_reverse(c,bl,boundary)
        assert np.allclose(kb,kr,rtol=1e-10,atol=1e-10)
        assert all(np.all(kb[int(t):,j]==0) for j,t in enumerate(boundary))
        report.update(forward_max_abs_error=error,adjoint_normalized_error=rel,duality_error=dualerr,
            upload_s=device.upload_seconds,old_forward_timing_s=ft,resident_forward_s=new['seconds'],
            old_adjoint_timing_s=rt,resident_adjoint=at,boundary_and_causality_passed=True)
        save('progress.json',dict(report,stage='gpu_equivalence_passed'))
        # Independent finite differences of NEURON, at fixed times (not derivative of argmin).
        weight_checks=[]
        for j in [0,len(loc)//2]:
            syn=m.synlist_i[int(oi[j])];original=syn.get_segment();weights=m.w_i.copy();last=None
            syn.loc(m.seglist[int(loc[j])])
            try:
                for eps in [1e-6,1e-7,1e-8]:
                    wp=weights.copy();wm=weights.copy();wp[oi[j]]+=eps;wm[oi[j]]-=eps
                    fd=(model.forward(m,wp)-model.forward(m,wm))/(2*eps)
                    scale=max(float(np.linalg.norm(fd)),1e-30)
                    row={'column':j,'epsilon_uS':eps,'relative_l2':float(np.linalg.norm(new['trace'][:,j]-fd)/scale),'fd_change':None if last is None else float(np.linalg.norm(fd-last)/scale)}
                    weight_checks.append(row);last=fd
            finally:
                syn.loc(original);m.w_i[:]=weights
                for syn,w in zip(m.synlist_i,weights):syn.w=float(w)
        report['weight_finite_difference']=weight_checks
        weight_pass=all(any(q['column']==j and q['relative_l2']<1e-3 and q['fd_change'] is not None and q['fd_change']<5e-3 for q in weight_checks) for j in [0,len(loc)//2])
        # Current-impulse test: outward +I is IClamp amplitude -I for one dt.
        current_checks=[];cpu_errors=[]
        choices=[int(j) for j in np.argsort(abs(direct))[-2:]]
        for j in choices:
            end=int(target[j]);assert end>2
            step=int(np.argmax(abs(kernel[:end,j])));local=int(loc[j])
            # Short exact CPU forward response from the impulse onward, zero initial tangent.
            short=dict(c,T=end-step)
            for name in ['vcoef','gcoef','diag','factor','vnext','wave']:short[name]=c[name][step:end].copy()
            short['wave']=np.zeros((end-step,1));short['wave'][0,0]=1.
            short['vnext'][:,local]=-69.
            cp=cpu_reference(short,[local],[0])['trace'][-1,0]
            ce=float(abs(cp-kernel[step,j])/max(1.,abs(cp)));cpu_errors.append(ce);assert ce<1e-10
            clamp=h.IClamp(m.seglist[local]);clamp.delay=step*h.dt;clamp.dur=h.dt;last=None
            for eps in [1e-5,1e-6,1e-7]:
                clamp.amp=-eps;plus=model.forward(m);clamp.amp=eps;minus=model.forward(m)
                fd=float((plus[end]-minus[end])/(2*eps));expected=float(kernel[step,j])
                current_checks.append({'column':j,'source_step':step,'target_step':end,'epsilon_nA':eps,'expected':expected,'finite_difference':fd,'relative_error':abs(fd-expected)/max(abs(expected),1e-30),'fd_change':None if last is None else abs(fd-last)/max(abs(fd),1e-30)})
                last=fd
            clamp.amp=0
        report['impulse_cpu_errors']=cpu_errors;report['current_finite_difference']=current_checks
        current_pass=all(any(q['column']==j and q['relative_error']<1e-3 and q['fd_change'] is not None and q['fd_change']<5e-3 for q in current_checks) for j in choices)
        save('progress.json',dict(report,stage='finite_difference_checked',weight_pass=weight_pass,current_pass=current_pass))
        assert weight_pass and current_pass, 'Independent NEURON finite-difference gate failed'
    # Exercise all 31 onset choices at 32 locations, write explicit-unit artifacts.
    report['profile_export']=make_profiles(m,c,dest/'pilot_profiles',identity,locations=loc)
    report['passed']=True;save('result.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':
    try:run()
    except BaseException:
        save('error.json',{'traceback':traceback.format_exc()});raise
