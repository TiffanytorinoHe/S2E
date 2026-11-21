from pathlib import Path
import ctypes
import numpy as np
from tangent import gpu

def adjoint(c,loc,targets):
    lib=ctypes.CDLL(str(Path(__file__).resolve().parent/'libadjoint.so'));fn=lib.adjoint_gpu;fn.restype=ctypes.c_int
    loc=np.asarray(loc,np.int32);targets=np.asarray(targets,np.int32);P=len(loc)
    assert np.all((targets>=0)&(targets<=c['T']))
    assert np.all((loc>=0)&(loc<c['n'])) and len(targets)==P
    order=np.argsort(targets,kind='stable');inverse=np.argsort(order)
    output=np.zeros((c['T'],P));timing=np.zeros(3)
    arrays=[c[k] for k in ['parent','order','lower','upper','segments','vcoef','gcoef','diag','D1']]+[loc[order],targets[order],output,timing]
    arrays=[np.ascontiguousarray(a) for a in arrays]
    args=[ctypes.c_int(x) for x in [c['n'],c['nn'],c['K'],c['T'],P,c['soma']]]+[ctypes.c_void_p(a.ctypes.data) for a in arrays]
    ret=fn(*args)
    if ret:raise RuntimeError(f'Adjoint CUDA error {ret}')
    return arrays[-2][:,inverse],arrays[-1].tolist()

def make_profiles(m,c,scores,dest):
    n=m.Nseg_no_axon
    best=np.argmin(scores.reshape(n,31),axis=1);ids=np.arange(n)*31+best
    loc=np.arange(n,dtype=np.int32);oi=best.astype(np.int32)
    assert m.meta['variant']=='pulse'
    trace=gpu(c,loc,oi,True)['trace'];targets=np.argmin(trace,axis=0).astype(np.int32)
    kernel,adjoint_timing=adjoint(c,loc,targets)
    outward=c['wave'][:,oi]*(c['vnext'][:,:n]+70.)
    reconstructed=np.sum(kernel*outward,axis=0)
    direct=trace[targets,np.arange(n)]
    normalized=np.max(abs(reconstructed-direct)/np.maximum(1.,abs(direct)))
    assert normalized<1e-8,f'Adjoint/forward duality failed: {normalized}'
    # Both forms are named with their units; no hidden dt or sign conventions.
    np.savez_compressed(dest/'profiles.npz',candidate_ids=ids,target_steps=targets,
        soma_gradient=trace,outward_current_per_uS=outward,
        kernel_discrete_mV_per_nA=kernel,continuous_kernel_multiplier=1/.025,
        zero_weight_waveforms=c['wave'][:,:m.probe_count],onset_grid_ms=m.meta['grid'],
        voltage=c['vnext'],time_ms=np.arange(1,c['T']+1)*.025,
        distance_um=m.distance[:n],D1=c['D1'][:n],dt_ms=.025)
    return {'segments':n,'duality_normalized_max_error':float(normalized),'adjoint_timing':adjoint_timing}
